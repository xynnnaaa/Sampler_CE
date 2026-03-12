// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <vector>
// #include <string>
// #include <random>
// #include <map>
// #include <unordered_set>

// namespace py = pybind11;

// // 用于存储 Beam 扩展中每条路径的状态
// struct PathState {
//     int t_idx;
//     py::dict vals;
//     py::object acc_bmp;
//     py::object rj_data;
//     py::object rj_bmp;
//     bool alive;
// };

// // 用于最终聚合比较的 Key
// struct BestKey {
//     int t_idx;
//     std::string rj_pk;
//     bool operator<(const BestKey& other) const {
//         if (t_idx != other.t_idx) return t_idx < other.t_idx;
//         return rj_pk < other.rj_pk;
//     }
// };

// // C++ 实现的采样核心逻辑
// py::tuple cpp_sample_beam_extensions(
//     py::object self_obj,          // 传入 Python 的 WanderJoinEngine 实例 (self)
//     py::list current_beam, 
//     py::list lookahead_plan, 
//     py::dict pid_map_full, 
//     py::dict global_map_full, 
//     int k_samples, 
//     std::string workload_name, 
//     py::object uncovered_mask_int // 传入 Python 大整数
// ) {
//     double execute_query_time = 0.0;
//     int batch_fetch_calls = 0;
//     double total_batch_fetch_time = 0.0;
//     int translate_calls = 0;
//     double total_translate_time = 0.0;

//     std::vector<PathState> active_paths;
//     active_paths.reserve(current_beam.size() * k_samples);

//     // 1. 初始化 active_paths
//     int t_idx = 0;
//     for (auto item_handle : current_beam) {
//         py::dict t_item = item_handle.cast<py::dict>();
//         py::dict t_data = t_item["data"].cast<py::dict>();
//         py::object t_bmp = t_item["bmp"];

//         for (int i = 0; i < k_samples; ++i) {
//             PathState path;
//             path.t_idx = t_idx;
//             path.vals = t_data.attr("copy")().cast<py::dict>();
//             path.acc_bmp = t_bmp;
//             path.rj_data = py::none();
//             path.rj_bmp = py::int_(0);
//             path.alive = true;
//             active_paths.push_back(path);
//         }
//         t_idx++;
//     }

//     // 随机数生成器 (用于 Wander Join)
//     std::random_device rd;
//     std::mt19937 gen(rd());

//     // 2. 执行 Plan
//     int step_idx = 0;
//     for (auto step_handle : lookahead_plan) {
//         py::dict step = step_handle.cast<py::dict>();
//         std::string alias = step["alias"].cast<std::string>();
//         std::string real_name = step["real_name"].cast<std::string>();
//         std::string parent_alias = step["parent"].cast<std::string>();
//         std::string raw_cond = step["join_condition"].cast<std::string>();
        
//         py::list sel_cols;
//         if (step.contains("sels")) {
//             sel_cols = step["sels"].cast<py::list>();
//         }

//         // 调用 Python 的 _parse_cond
//         py::tuple parsed = self_obj.attr("_parse_cond")(raw_cond, alias, parent_alias).cast<py::tuple>();
//         if (parsed[0].is_none()) { step_idx++; continue; }
        
//         std::string my_col = parsed[0].cast<std::string>();
//         std::string parent_col = parsed[1].cast<std::string>();
//         std::string parent_key = parent_alias + "." + parent_col;

//         py::list batch_vals;
//         for (auto& path : active_paths) {
//             if (path.alive) {
//                 py::object val = path.vals.attr("get")(parent_key);
//                 if (!val.is_none()) {
//                     std::string val_str = py::str(val);
//                     if (val_str != "None") {
//                         batch_vals.append(val);
//                         continue;
//                     }
//                 }
//                 path.alive = false;
//             }
//         }

//         if (py::len(batch_vals) == 0) break;

//         // 调用 Python 的 _batch_fetch_neighbors
//         auto t_bf0 = std::chrono::high_resolution_clock::now();
//         py::tuple bf_res = self_obj.attr("_batch_fetch_neighbors")(real_name, my_col, batch_vals, sel_cols, alias).cast<py::tuple>();
//         auto t_bf1 = std::chrono::high_resolution_clock::now();
        
//         py::dict neighbors = bf_res[0].cast<py::dict>();
//         double execute_time = bf_res[1].cast<double>();
        
//         batch_fetch_calls++;
//         total_batch_fetch_time += std::chrono::duration<double>(t_bf1 - t_bf0).count();
//         execute_query_time += execute_time;

//         py::dict my_pid_map = pid_map_full.attr("get")(alias, py::dict()).cast<py::dict>();
//         py::object my_global_mask = global_map_full.attr("get")(alias, py::int_(0));

//         std::unordered_set<std::string> pending_bitmap_ids;
//         std::map<int, py::dict> path_selections;

//         int wander_success_count = 0;
//         for (size_t i = 0; i < active_paths.size(); ++i) {
//             if (!active_paths[i].alive) continue;

//             py::object p_val_obj = active_paths[i].vals.attr("get")(parent_key);
//             std::string p_val = py::str(p_val_obj);

//             if (!neighbors.contains(p_val)) {
//                 active_paths[i].alive = false;
//                 continue;
//             }

//             py::list candidates = neighbors[p_val.c_str()].cast<py::list>();
//             if (py::len(candidates) == 0) {
//                 active_paths[i].alive = false;
//                 continue;
//             }

//             // Wander Join: 随机挑选
//             std::uniform_int_distribution<> dis(0, py::len(candidates) - 1);
//             py::dict chosen = candidates[dis(gen)].cast<py::dict>();
//             path_selections[i] = chosen;

//             std::string alias_id_key = alias + ".id";
//             py::object chosen_id = chosen.attr("get")(alias_id_key);
            
//             if (!chosen_id.is_none()) {
//                 pending_bitmap_ids.insert(py::str(chosen_id));
//             } else {
//                 active_paths[i].alive = false;
//             }
//         }

//         if (pending_bitmap_ids.empty()) { step_idx++; continue; }

//         py::list pending_ids_list;
//         for (const auto& id : pending_bitmap_ids) {
//             pending_ids_list.append(id);
//         }

//         // 调用 Python 的 _batch_fetch_translate_bitmaps
//         auto t_tr0 = std::chrono::high_resolution_clock::now();
//         py::tuple tr_res = self_obj.attr("_batch_fetch_translate_bitmaps")(
//             real_name, pending_ids_list, workload_name, alias, my_pid_map, my_global_mask
//         ).cast<py::tuple>();
//         auto t_tr1 = std::chrono::high_resolution_clock::now();

//         py::dict translated_map = tr_res[0].cast<py::dict>();
//         execute_time = tr_res[1].cast<double>();

//         translate_calls++;
//         total_translate_time += std::chrono::duration<double>(t_tr1 - t_tr0).count();
//         execute_query_time += execute_time;

//         // 更新 paths
//         for (auto const& [i, chosen] : path_selections) {
//             std::string alias_id_key = alias + ".id";
//             py::object chosen_id = chosen.attr("get")(alias_id_key);
//             py::object qid_mask = translated_map.attr("get")(chosen_id, my_global_mask);

//             // 位运算 (大整数)
//             active_paths[i].acc_bmp = active_paths[i].acc_bmp.attr("__and__")(qid_mask);
//             active_paths[i].vals.attr("update")(chosen);

//             if (step_idx == 0) {
//                 active_paths[i].rj_data = chosen;
//                 active_paths[i].rj_bmp = qid_mask;
//             }
//         }
//         step_idx++;
//     }

//     // 3. 聚合结果
//     py::list proposed_candidates;
//     std::map<BestKey, py::dict> best_results;
    
//     std::string rj_alias = lookahead_plan[0].cast<py::dict>()["alias"].cast<std::string>();
//     std::string rj_pk_key = rj_alias + ".id";
//     int wander_success_count = 0;

//     for (const auto& path : active_paths) {
//         if (path.alive && !path.rj_data.is_none()) {
//             wander_success_count++;
//             py::dict rj_data_dict = path.rj_data.cast<py::dict>();
//             std::string rj_pk = py::str(rj_data_dict[rj_pk_key.c_str()]);
//             BestKey key{path.t_idx, rj_pk};

//             // 计算 Score
//             py::object and_result = path.acc_bmp.attr("__and__")(uncovered_mask_int);
//             int score = 0;
//             // 兼容 Python 3.10+ 的 bit_count
//             if (py::hasattr(and_result, "bit_count")) {
//                 score = and_result.attr("bit_count")().cast<int>();
//             } else {
//                 py::object builtins = py::module_::import("builtins");
//                 std::string bin_str = py::str(builtins.attr("bin")(and_result));
//                 score = std::count(bin_str.begin(), bin_str.end(), '1');
//             }

//             if (best_results.find(key) == best_results.end() || score > best_results[key]["score"].cast<int>()) {
//                 py::dict res;
//                 res["score"] = score;
//                 res["rj_data"] = path.rj_data;
//                 res["rj_bmp"] = path.rj_bmp;
//                 best_results[key] = res;
//             }
//         }
//     }

//     for (const auto& [key, res] : best_results) {
//         py::dict cand;
//         cand["t_idx"] = key.t_idx;
//         cand["rj_data"] = res["rj_data"];
//         cand["rj_bmp"] = res["rj_bmp"];
//         cand["score"] = res["score"];
//         proposed_candidates.append(cand);
//     }

//     // 在 C++ 中不方便直接做 random.shuffle（因为是 py::list），我们将留到 Python 端打乱
//     py::print("                _batch_fetch_neighbors called ", batch_fetch_calls, " times, total time: ", total_batch_fetch_time, "s");
//     py::print("                _translate_pid_bitmap called ", translate_calls, " times, total time: ", total_translate_time, "s");
    
//     double success_rate = active_paths.empty() ? 0.0 : (double)wander_success_count / active_paths.size();
//     py::print("                wander join success count: ", wander_success_count, ", success rate: ", success_rate);

//     return py::make_tuple(proposed_candidates, execute_query_time);
// }

// PYBIND11_MODULE(cpp_wander_join, m) {
//     m.def("cpp_sample_beam_extensions", &cpp_sample_beam_extensions, "C++ optimized sample_beam_extensions");
// }

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <random>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <Python.h> // 引入底层 CPython API 以获取极限性能

namespace py = pybind11;

// V2 优化：将 vals 彻底变为纯 C++ map，查找速度提升百倍
struct PathState {
    int t_idx;
    std::unordered_map<std::string, std::string> vals; 
    py::object acc_bmp;
    py::object rj_data; // 留着最终输出用
    py::object rj_bmp;
    bool alive;
};

struct BestKey {
    int t_idx;
    std::string rj_pk;
    bool operator<(const BestKey& other) const {
        if (t_idx != other.t_idx) return t_idx < other.t_idx;
        return rj_pk < other.rj_pk;
    }
};

py::tuple cpp_sample_beam_extensions(
    py::object self_obj, 
    py::list current_beam, 
    py::list lookahead_plan, 
    py::dict pid_map_full, 
    py::dict global_map_full, 
    int k_samples, 
    std::string workload_name, 
    py::object uncovered_mask_int 
) {
    double execute_query_time = 0.0;
    int batch_fetch_calls = 0;
    double total_batch_fetch_time = 0.0;
    int translate_calls = 0;
    double total_translate_time = 0.0;

    std::vector<PathState> active_paths;
    active_paths.reserve(current_beam.size() * k_samples);

    // 1. 初始化 active_paths，直接将 Python Dict 扁平化为 C++ Map
    int t_idx = 0;
    for (auto item_handle : current_beam) {
        py::dict t_item = item_handle.cast<py::dict>();
        py::dict t_data = t_item["data"].cast<py::dict>();
        py::object t_bmp = t_item["bmp"];

        // 提前解析一次 python dict
        std::unordered_map<std::string, std::string> cpp_t_data;
        for (auto kv : t_data) {
            cpp_t_data[kv.first.cast<std::string>()] = py::str(kv.second).cast<std::string>();
        }

        for (int i = 0; i < k_samples; ++i) {
            PathState path;
            path.t_idx = t_idx;
            path.vals = cpp_t_data; // C++ native copy，极快
            path.acc_bmp = t_bmp;
            path.rj_data = py::none();
            path.rj_bmp = py::int_(0);
            path.alive = true;
            active_paths.push_back(path);
        }
        t_idx++;
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    int step_idx = 0;
    for (auto step_handle : lookahead_plan) {
        py::dict step = step_handle.cast<py::dict>();
        std::string alias = step["alias"].cast<std::string>();
        std::string real_name = step["real_name"].cast<std::string>();
        std::string parent_alias = step["parent"].cast<std::string>();
        std::string raw_cond = step["join_condition"].cast<std::string>();
        
        py::list sel_cols;
        if (step.contains("sels")) {
            sel_cols = step["sels"].cast<py::list>();
        }

        py::tuple parsed = self_obj.attr("_parse_cond")(raw_cond, alias, parent_alias).cast<py::tuple>();
        if (parsed[0].is_none()) { step_idx++; continue; }
        
        std::string my_col = parsed[0].cast<std::string>();
        std::string parent_col = parsed[1].cast<std::string>();
        std::string parent_key = parent_alias + "." + parent_col;

        // V2 优化：使用 std::unordered_set 提前去重，减轻 Python 端的压力
        std::unordered_set<std::string> deduplicated_vals;
        for (auto& path : active_paths) {
            if (path.alive) {
                auto it = path.vals.find(parent_key);
                if (it != path.vals.end() && it->second != "None") {
                    deduplicated_vals.insert(it->second);
                } else {
                    path.alive = false;
                }
            }
        }

        if (deduplicated_vals.empty()) break;

        py::list batch_vals;
        for (const auto& val : deduplicated_vals) {
            batch_vals.append(val);
        }

        auto t_bf0 = std::chrono::high_resolution_clock::now();
        py::tuple bf_res = self_obj.attr("_batch_fetch_neighbors")(real_name, my_col, batch_vals, sel_cols, alias).cast<py::tuple>();
        auto t_bf1 = std::chrono::high_resolution_clock::now();
        
        py::dict neighbors = bf_res[0].cast<py::dict>();
        double execute_time = bf_res[1].cast<double>();
        
        batch_fetch_calls++;
        total_batch_fetch_time += std::chrono::duration<double>(t_bf1 - t_bf0).count();
        execute_query_time += execute_time;

        py::dict my_pid_map = pid_map_full.contains(alias) ? pid_map_full[alias.c_str()].cast<py::dict>() : py::dict();
        py::object my_global_mask = global_map_full.contains(alias) ? global_map_full[alias.c_str()] : py::int_(0);

        std::unordered_set<std::string> pending_bitmap_ids;
        std::map<int, py::dict> path_selections;

        std::string alias_id_key = alias + ".id";

        for (size_t i = 0; i < active_paths.size(); ++i) {
            if (!active_paths[i].alive) continue;

            const std::string& p_val = active_paths[i].vals[parent_key];

            if (!neighbors.contains(p_val)) {
                active_paths[i].alive = false;
                continue;
            }

            py::list candidates = neighbors[p_val.c_str()].cast<py::list>();
            if (py::len(candidates) == 0) {
                active_paths[i].alive = false;
                continue;
            }

            std::uniform_int_distribution<> dis(0, py::len(candidates) - 1);
            py::dict chosen = candidates[dis(gen)].cast<py::dict>();
            path_selections[i] = chosen;

            if (chosen.contains(alias_id_key)) {
                py::object chosen_id = chosen[alias_id_key.c_str()];
                if (!chosen_id.is_none()) {
                    pending_bitmap_ids.insert(py::str(chosen_id));
                } else {
                    active_paths[i].alive = false;
                }
            } else {
                active_paths[i].alive = false;
            }
        }

        if (pending_bitmap_ids.empty()) { step_idx++; continue; }

        py::list pending_ids_list;
        for (const auto& id : pending_bitmap_ids) {
            pending_ids_list.append(id);
        }

        auto t_tr0 = std::chrono::high_resolution_clock::now();
        py::tuple tr_res = self_obj.attr("_batch_fetch_translate_bitmaps")(
            real_name, pending_ids_list, workload_name, alias, my_pid_map, my_global_mask
        ).cast<py::tuple>();
        auto t_tr1 = std::chrono::high_resolution_clock::now();

        py::dict translated_map = tr_res[0].cast<py::dict>();
        execute_time = tr_res[1].cast<double>();

        translate_calls++;
        total_translate_time += std::chrono::duration<double>(t_tr1 - t_tr0).count();
        execute_query_time += execute_time;

        // 更新 paths
        for (auto const& [i, chosen] : path_selections) {
            py::object chosen_id = chosen[alias_id_key.c_str()];
            py::object qid_mask = translated_map.contains(chosen_id) ? translated_map[chosen_id] : my_global_mask;

            // V2 优化：使用 C-API PyNumber_And 替代 .attr("__and__")，极其高效
            PyObject* and_res = PyNumber_And(active_paths[i].acc_bmp.ptr(), qid_mask.ptr());
            active_paths[i].acc_bmp = py::reinterpret_steal<py::object>(and_res);

            // V2 优化：将 Python Dict 更新到 C++ Map 中
            for (auto kv : chosen) {
                active_paths[i].vals[kv.first.cast<std::string>()] = py::str(kv.second).cast<std::string>();
            }

            if (step_idx == 0) {
                active_paths[i].rj_data = chosen;
                active_paths[i].rj_bmp = qid_mask;
            }
        }
        step_idx++;
    }

    py::list proposed_candidates;
    std::map<BestKey, py::dict> best_results;
    
    std::string rj_alias = lookahead_plan[0].cast<py::dict>()["alias"].cast<std::string>();
    std::string rj_pk_key = rj_alias + ".id";
    int wander_success_count = 0;

    for (const auto& path : active_paths) {
        if (path.alive && !path.rj_data.is_none()) {
            wander_success_count++;
            py::dict rj_data_dict = path.rj_data.cast<py::dict>();
            std::string rj_pk = py::str(rj_data_dict[rj_pk_key.c_str()]);
            BestKey key{path.t_idx, rj_pk};

            PyObject* and_result = PyNumber_And(path.acc_bmp.ptr(), uncovered_mask_int.ptr());
            py::object and_obj = py::reinterpret_steal<py::object>(and_result);

            int score = 0;
            // V2 优化：使用 C-API 极速调用 bit_count
            if (PyObject_HasAttrString(and_obj.ptr(), "bit_count")) {
                PyObject* count_obj = PyObject_CallMethod(and_obj.ptr(), "bit_count", nullptr);
                score = PyLong_AsLong(count_obj);
                Py_DECREF(count_obj);
            } else {
                py::object builtins = py::module_::import("builtins");
                std::string bin_str = py::str(builtins.attr("bin")(and_obj));
                score = std::count(bin_str.begin(), bin_str.end(), '1');
            }

            if (best_results.find(key) == best_results.end() || score > best_results[key]["score"].cast<int>()) {
                py::dict res;
                res["score"] = score;
                res["rj_data"] = path.rj_data;
                res["rj_bmp"] = path.rj_bmp;
                best_results[key] = res;
            }
        }
    }

    for (const auto& [key, res] : best_results) {
        py::dict cand;
        cand["t_idx"] = key.t_idx;
        cand["rj_data"] = res["rj_data"];
        cand["rj_bmp"] = res["rj_bmp"];
        cand["score"] = res["score"];
        proposed_candidates.append(cand);
    }

    py::print("                _batch_fetch_neighbors called ", batch_fetch_calls, " times, total time: ", total_batch_fetch_time, "s");
    py::print("                _translate_pid_bitmap called ", translate_calls, " times, total time: ", total_translate_time, "s");
    
    double success_rate = active_paths.empty() ? 0.0 : (double)wander_success_count / active_paths.size();
    py::print("                wander join success count: ", wander_success_count, ", success rate: ", success_rate);

    return py::make_tuple(proposed_candidates, execute_query_time);
}

PYBIND11_MODULE(cpp_wander_join, m) {
    m.def("cpp_sample_beam_extensions", &cpp_sample_beam_extensions, "C++ optimized sample_beam_extensions V2");
}