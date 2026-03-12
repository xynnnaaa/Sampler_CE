from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "cpp_wander_join",
        ["cpp_wander_join.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"], # 开启最高级别优化
    ),
]

setup(
    name="cpp_wander_join",
    ext_modules=ext_modules,
)