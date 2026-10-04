"""Regression tests for identifier normalization without changing SQL values."""
import unittest
from pathlib import Path
import sys

# Match the sampler's path setup for the project's legacy absolute imports.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from mscn.query_representation.utils import extract_join_graph


class IdentifierCaseTests(unittest.TestCase):
    def test_preserve_tpch_literal_values(self):
        graph = extract_join_graph("""
            SELECT COUNT(*) FROM part TPCH_P, customer TPCH_C
            WHERE TPCH_P.p_partkey = TPCH_C.c_custkey
              AND TPCH_P.p_brand = 'Brand#35'
              AND TPCH_C.c_mktsegment = 'MACHINERY'
        """)
        self.assertEqual(graph.nodes['tpch_p']['predicates'], ["tpch_p.p_brand = 'Brand#35'"])
        self.assertEqual(graph.nodes['tpch_c']['predicates'], ["tpch_c.c_mktsegment = 'MACHINERY'"])

    def test_mixed_case_alias_references_match_nodes(self):
        graph = extract_join_graph("""
            SELECT COUNT(*) FROM PART AS TPCH_P, PARTSUPP AS TPCH_PS
            WHERE TpCh_P.P_PARTKEY = tpch_ps.PS_PARTKEY
              AND tpch_P.P_BRAND = 'Brand#35'
        """)
        self.assertEqual(set(graph.nodes), {'tpch_p', 'tpch_ps'})
        self.assertEqual(graph.nodes['tpch_p']['real_name'], 'part')
        self.assertEqual(graph.nodes['tpch_ps']['real_name'], 'partsupp')
        self.assertEqual(graph['tpch_p']['tpch_ps']['join_condition'],
                         'tpch_p.p_partkey = tpch_ps.ps_partkey')
        self.assertEqual(graph.nodes['tpch_p']['predicates'], ["tpch_p.p_brand = 'Brand#35'"])

    def test_quoted_identifiers_keep_case(self):
        graph = extract_join_graph("""
            SELECT COUNT(*) FROM "Part" AS "P"
            WHERE "P"."Brand" = 'Brand#35'
        """)
        self.assertEqual(set(graph.nodes), {'P'})
        self.assertEqual(graph.nodes['P']['real_name'], 'Part')
        self.assertEqual(graph.nodes['P']['predicates'], ['"P"."Brand" = \'Brand#35\''])

    def test_literals_with_escaped_quotes_and_alias_text(self):
        graph = extract_join_graph("""
            SELECT COUNT(*) FROM PART AS P
            WHERE p.p_name = 'O''Brien P.Value'
              AND P.p_brand IN ('Brand#35', 'Brand#43')
              AND p.p_type LIKE 'PROMO%'
        """)
        predicates = graph.nodes['p']['predicates']
        self.assertIn("p.p_name = 'O''Brien P.Value'", predicates)
        self.assertIn("p.p_brand IN ('Brand#35', 'Brand#43')", predicates)
        self.assertIn("p.p_type LIKE 'PROMO%'", predicates)

    def test_numeric_filters_and_no_predicate_alias_unchanged(self):
        graph = extract_join_graph("""
            SELECT COUNT(*) FROM LINEITEM LI, ORDERS O
            WHERE li.L_ORDERKEY = o.O_ORDERKEY
              AND LI.l_tax = 0.02 AND li.l_partkey >= 119472
              AND li.l_quantity < 25
        """)
        self.assertEqual(set(graph.nodes['li']['predicates']),
                         {'li.l_tax = 0.02', 'li.l_partkey >= 119472', 'li.l_quantity < 25'})
        self.assertEqual(graph.nodes['o']['predicates'], [])


if __name__ == '__main__':
    unittest.main()
