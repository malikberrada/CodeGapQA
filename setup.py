from __future__ import annotations

import os
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CodeGapBuildExt(build_ext):
    """Build the native backend when a compatible compiler is available.

    Set CODEGAP_REQUIRE_NATIVE=1 to turn any compilation failure into a hard
    installation error. The default keeps the pure-Python fallback installable.
    """

    def _required(self) -> bool:
        return os.environ.get("CODEGAP_REQUIRE_NATIVE", "0") == "1"

    def build_extensions(self) -> None:
        compiler = self.compiler.compiler_type
        disable_openmp = os.environ.get("CODEGAP_DISABLE_OPENMP", "0") == "1"
        for extension in self.extensions:
            if compiler == "msvc":
                extension.extra_compile_args = ["/O2", "/std:c++17", "/EHsc"]
                if not disable_openmp:
                    extension.extra_compile_args.append("/openmp")
            else:
                extension.extra_compile_args = ["-O3", "-std=c++17"]
                if not disable_openmp and sys.platform != "darwin":
                    extension.extra_compile_args.append("-fopenmp")
                    extension.extra_link_args = ["-fopenmp"]
        try:
            super().build_extensions()
        except Exception as exception:
            if self._required():
                raise
            self.warn(
                "CodeGap-QA native backend was not built; installing the "
                f"reference Python fallback. Cause: {exception}"
            )

    def build_extension(self, extension) -> None:
        try:
            super().build_extension(extension)
        except Exception as exception:
            if self._required():
                raise
            self.warn(
                f"Skipping optional native extension {extension.name}: {exception}"
            )


setup(
    ext_modules=[
        Extension(
            "codegap_qa._fast_cpu",
            sources=["native/fast_cpu.cpp"],
            language="c++",
            optional=True,
        )
    ],
    cmdclass={"build_ext": CodeGapBuildExt},
)
