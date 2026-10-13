# numsol

The array layer for [Saule](https://github.com/lauriszz123/saule) — numpy's
shape, in Saule's types, using Saule's own operators.

```saule
import * from "numsol"

local a: NDArray = Num.arange(0.0, 6.0).reshape({2, 3})
local b: NDArray = Num.ones({2, 3})

println(a * 2.0 + b)              -- elementwise, broadcasting
println(a.matmul(b.transpose()))  -- (2, 3) @ (3, 2) -> (2, 2)
println(a.sumAxis(1, true))       -- one row of column sums
```

It is a pure Saule package: a git repo with a `saule.config`, no native
code, nothing to build.

## Install

```
dependencies: ["../saule-numsol"]
```

The package name is `numsol`, so `import * from "numsol"`.

## Operators

Arithmetic is the language's, not a method set bolted on top. The
right-hand side may be another array, a lone `float`, or a `table<float>`:

| Expression | Meaning |
|---|---|
| `a + b` `a - b` `a * b` `a / b` | elementwise, with broadcasting |
| `a % b` `a ^ b` | elementwise remainder and power |
| `a + 2.0` `a * 2.0` `a ^ 2.0` | a scalar broadcasts over every element |
| `a + {1.0, 2.0}` | a float table is read as a vector |
| `-a` | negated |
| `#a` | element count |
| `a[2]` / `a[2] = 9.0` | flat element, 1-based, read and write |
| `a == b` | same shape, same contents |
| `local v: NDArray = {1.0, 2.0}` | a table literal fills an array slot |

Two things to keep in mind:

- **Arithmetic dispatches on the left operand**, so `2.0 * a` cannot work.
  Put the array first.
- **`*` is elementwise.** The matrix product is `a.matmul(b)` — Saule has no
  `@` for it to borrow.

## Conventions

**Indices are 1-based.** `a.get({1, 1})` is the first element and `a.row(1)`
the first row, matching `String.sub`, `Table.insert` and `for i, v in t`.
Carrying numpy's 0-based habit into a 1-based language buys nothing and
costs every off-by-one twice.

**Elements are `float`.** Saule never promotes `integer` to `float`, and the
library does not get a private exemption — `a + 2` is an error that names
the fix rather than a silent conversion. Write `2.0` or `2f`.

**Arrays are contiguous and never aliased.** There are no views: `reshape`,
`transpose` and `slice` all return fresh arrays. That costs a copy and buys
the guarantee that writing through one name cannot surprise another. The
in-place family — `addInPlace`, `scaleInPlace`, `fill`, `a[i] = v` — is the
deliberate exception, for accumulators and optimiser inner loops.

## What's in it

| Name | Role |
|---|---|
| `Num` | where arrays are made: `zeros` `ones` `full` `eye` `arange` `linspace` `random` `uniform` `randn` `array` `matrix` `shaped` `vstack`, plus function forms of every method |
| `NDArray` | the array: operators, `reshape` `transpose` `row` `col` `slice` `broadcastTo`, `map` `zip`, `exp` `log` `sqrt` `abs` `tanh` `sigmoid` `step` `maximum` `clamp`, `sum` `mean` `argmax` `norm`, `sumAxis` `meanAxis` `maxAxis` `argmaxAxis`, `matmul` `dot`, `allClose` |
| `Shape` | index arithmetic over `table<integer>` — sizes, broadcasting, flat offsets |
| `Cursor` | an odometer over a shape, for walking one element at a time |
| `Scalar` | float-level maths `Math` lacks: `tanh` `sigmoid` `step` |
| `Kernel` | the loops, over flat `table<float>` buffers |
| `Format` | how an array renders |

The layering is one-way: `Kernel` and `Shape` know nothing about `NDArray`,
which is why a native backend would replace `kernel.sau` and leave
everything else alone.

## Broadcasting

numpy's rules exactly. Axes line up from the **right**; each pair must be
equal, or one of them must be 1 and gets stretched.

```saule
local m: NDArray = Num.matrix({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}})

m + Num.matrix({{10.0, 20.0, 30.0}})   -- (2,3) + (1,3): row reused
m + Num.array({1.0, 1.0, 1.0})         -- (2,3) + (3):   aligned right
m + Num.matrix({{1.0}, {2.0}})         -- (2,3) + (2,1): column reused
m + 2.0                                -- a scalar reaches everywhere
```

## Example

```sh
cd examples/tour && saule run
```

A guided tour of the whole API where every claim is asserted — 105 checks
covering construction, shapes, operators, indexing, broadcasting,
reductions and linear algebra. It is the test suite and the documentation
at once.

## Known limits

- **`matmul` is rank 2.** Batched (rank 3+) products are not implemented;
  rank-1 operands are promoted and demoted the way numpy's `@` does.
- **No views or strides.** Every reshape copies. Fine at the sizes Saule
  runs today, and the first thing to revisit if that changes.
- **No integer arrays.** One element type, `float`.
- **Pure Saule, so interpreter speed.** The kernels are plain loops on a
  register VM; `kernel.sau` is the single file a native backend replaces.

## Built with

[zenith](../saule-zenith) — neural networks on top of these arrays.
