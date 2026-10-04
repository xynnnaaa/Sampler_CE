"""Compare actual sampler methods with V1, with shared predicate caching enabled."""
import ast
from collections import defaultdict
from contextlib import redirect_stdout
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
import unittest

NEW=Path(__file__).resolve().parents[1]
OLD=NEW.parent/'new_model'
sys.path.insert(0,str(NEW))
from template_annotations import TemplateAnnotations, AliasPlan, AliasBitmaps, SharedMemoryBudget
from sampling_timing import SamplingTimings, timed, timed_template
from predicate_cache import SharedPredicateCache
import wander_join as v2_engine
spec=importlib.util.spec_from_file_location('old_timing',OLD/'sampling_timing.py')
old_timing=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_timing)


def class_from_file(path,name,v2=False):
    tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
    prof=sys.modules['sampling_timing'] if v2 else old_timing
    namespace=dict(defaultdict=defaultdict,random=random,re=re,time=time,
        SamplingTimings=prof.SamplingTimings,timed=prof.timed,timed_template=prof.timed_template,
        TemplateAnnotations=TemplateAnnotations,SharedMemoryBudget=SharedMemoryBudget)
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace[name]


class Graph:
    def __init__(self,aliases):self.aliases=aliases
    def nodes(self,data=False):return list(self.aliases.items()) if data else list(self.aliases)


class Cursor:
    def __init__(self,conn,name=None):
        self.conn=conn;self.name=name;self.closed=False;self.index=0
    def __enter__(self):return self
    def __exit__(self,*args):self.closed=True
    def execute(self,sql):
        if self.conn.fail and (self.name or 'COUNT(*)' in sql):raise RuntimeError('injected build failure')
        self.conn.queries.append(sql)
        if self.name:
            self.conn.stream_queries.append(sql)
        table=re.search(r'FROM\s+(?:"([^"]+)"|(\w+))',sql).group(1) or re.search(r'FROM\s+(?:"([^"]+)"|(\w+))',sql).group(2)
        if 'COUNT(*)' in sql:
            rows=self.conn.data[table];ids=[r['id'] for r in rows]
            self.rows=[(len(rows),min(ids) if ids else None,max(ids) if ids else None)];return
        if self.name:
            width=int(re.search(r'bit varying\((\d+)\)',sql)[1]) if 'bit varying' in sql else len(re.search(r"B'([01]+)'",sql)[1])
            global_mask=int(re.search(r"B'([01]+)'",sql)[1],2)
            cases=re.findall(r"CASE WHEN \((.*?)\) THEN B'([01]+)' ELSE B'([01]+)' END",sql)
            self.rows=[]
            for row in (sorted(self.conn.data[table],key=lambda r:r['id']) if 'ORDER BY id' in sql else self.conn.data[table]):
                mask=global_mask
                for pred,yes,no in cases:
                    if self.conn.matches(pred,row):mask |= int(yes,2)
                self.rows.append((row['id'],format(mask,f'0{width}b')))
            return
        self.conn.neighbor_queries.append(sql)
        if 'ORDER BY RANDOM()' in sql:
            self.rows=[(r['id'],) for r in self.conn.data[table]];return
        ids={int(s) for s in re.search(r'IN \(([^)]+)\)',sql)[1].replace("'",'').split(',')}
        if '_anno_idx' in table:
            table=table.split('_anno_idx')[0]
            self.rows=[(r['id'], ('10' if r['id']%2 else '01')+'0'*6) for r in self.conn.data[table] if r['id'] in ids]
        else:
            col=re.search(r'WHERE t\.(\w+)',sql)[1];cols=re.findall(r't\.(\w+)',sql.split('FROM')[0])
            self.rows=[tuple(r[c] for c in cols) for r in self.conn.data[table] if r[col] in ids and all(r[c] is not None for c in cols)]
    def fetchone(self):return self.rows[0]
    def fetchall(self):
        if self.name:raise AssertionError('Server-side stream must not fetchall')
        return self.rows
    def fetchmany(self,size):
        self.conn.batch_sizes.append(size)
        result=self.rows[self.index:self.index+size];self.index+=len(result);return result


class Connection:
    def __init__(self,data):
        self.data=data;self.queries=[];self.neighbor_queries=[];self.stream_queries=[];self.cursors=[];self.batch_sizes=[];self.fail=False
    def cursor(self,name=None):
        c=Cursor(self,name);self.cursors.append(c);return c
    @staticmethod
    def matches(pred,row):
        if pred=='id % 2 = 1':return row['id']%2==1
        if pred=='id % 2 = 0':return row['id']%2==0
        if pred=='id < 0':return row['id']<0
        if pred=='flag = 1':return row.get('flag')==1
        if pred=='id > -999':return row['id']>-999
        raise AssertionError(pred)


TEST_TEMPORARIES=[]

def fixture(v2):
    typ=class_from_file((NEW if v2 else OLD)/'join_sampler_linear.py','JoinSampler',v2)
    engine=v2_engine.WanderJoinEngine if v2 else class_from_file(OLD/'wander_join.py','WanderJoinEngine')
    obj=typ.__new__(typ)
    obj.conn=Connection({
        'root':[dict(id=i,key=i) for i in range(1,7)],
        'child':[dict(id=i*10+j,fk=i,key=i*10+j) for i in (1,2,4,5,6) for j in (0,1)],
        'leaf':[dict(id=i+100,fk=i) for i in (10,11,20,40,41,50,51,60,61)]})
    obj.cursor=obj.conn.cursor();obj.engine=engine(obj.conn,obj.cursor);obj.timings=obj.engine.timings
    obj.m_partitions,obj.k_bitmaps,obj.w_samples=3,2,3;obj.workload_name='test'
    obj.global_pid_to_pred={t:{0:'id % 2 = 1',1:'id % 2 = 0',7:'id < 0'} for t in obj.conn.data}
    obj.annotation_batch_size=2;obj.annotation_budget=None;obj.annotation_table_stats_cache={};obj.annotation_predicate_cache=None;obj.annotation_compose_batch_size=3
    if v2:
        obj.test_cache_temp=tempfile.TemporaryDirectory()
        TEST_TEMPORARIES.append(obj.test_cache_temp)
        obj.annotation_predicate_cache=SharedPredicateCache(obj.test_cache_temp.name,'sample-regression',{},obj.global_pid_to_pred,obj.timings,batch_size=2)
    obj.add_sel_info_to_graph=lambda graph:None
    plan=[dict(alias='a',real_name='root',parent=None,join_condition=None,sels=['a.id','a.key']),
          dict(alias='b',real_name='child',parent='a',join_condition='a.key=b.fk',sels=['b.id','b.fk','b.key']),
          dict(alias='c',real_name='leaf',parent='b',join_condition='b.key=c.fk',sels=['c.id','c.fk'])]
    obj.build_join_tree_structure=lambda graph,aliases:plan
    return obj

KEY=(('a','b','c'),'a.key=b.fk||b.key=c.fk')
GRAPH=Graph({a:{'real_name':t} for a,t in [('a','root'),('b','child'),('c','leaf')]})
DATA=dict(graph=GRAPH,instances=[dict(a=0,b=-1,c=0),dict(a=1,b=0,c=-1),dict(a=7,b=7,c=7)])



class SamplingEquivalence(unittest.TestCase):
    def tearDown(self):
        for temporary in TEST_TEMPORARIES:
            temporary.cleanup()
        TEST_TEMPORARIES.clear()

    def test_equivalence_to_v1_samples_and_neighbor_queries(self):
        for seed in range(30):
            old,new=fixture(False),fixture(True)
            with redirect_stdout(io.StringIO()):
                random.seed(seed);before=old.sample_for_one_template(KEY,DATA)
                before_random_state = random.getstate()
                random.seed(seed);after=new.sample_for_one_template(KEY,DATA)
                after_random_state = random.getstate()
            self.assertEqual(before,after)
            self.assertEqual(before_random_state, after_random_state)
            def neighbor_only(conn):return [q for q in conn.neighbor_queries if '_anno_idx' not in q]
            self.assertEqual(neighbor_only(old.conn),neighbor_only(new.conn))
            for k in old.timings.counters:
                if k.startswith(('root.','wander_join.','samples.','random_walk.','partitions.')):
                    self.assertEqual(old.timings.counters[k],new.timings.counters[k])
            self.assertIsNone(new.engine.annotations)
            self.assertEqual(new.timings._stack,[])
            self.assertFalse(any('_anno_idx' in q for q in new.conn.queries))
            self.assertEqual(new.conn.batch_sizes,[2]*len(new.conn.batch_sizes))


if __name__ == '__main__':
    unittest.main()
