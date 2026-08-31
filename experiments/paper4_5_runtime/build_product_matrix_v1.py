"""Deprecated compatibility import for the product-matrix v2 builder."""

from experiments.paper4_5_runtime.build_product_matrix_v2 import (  # noqa: F401
    DEFAULT_OUTPUT,
    DEFAULT_TABLE,
    build_matrix,
    main,
    write_table,
)


if __name__ == "__main__":
    main()
