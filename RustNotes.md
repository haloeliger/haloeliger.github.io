# On Elementary Matrix Computations in Rust #
### by Hans-Andrea Loeliger ###

*Version 0.1, Oct. 5, 2026*

Like many other programmers, I am a big fan of Rust. 
However, learning Rust and using it properly is not so easy.

In this note, I discuss 
elementary matrix-times-vector and matrix-times-vector multiplications,
which are the computational backbone of most scientific computing and neural networks.

Many thanks go to Hampus Malmberg, Hugo Aguettaz, and Cyrill Achermann,
who helped me a lot to get started with Rust.


- [For Loops and Iterators](#for-loops-and-iterators)
- [BLAS and OxiBLAS](#blas-and-oxiblas)
- [Abstract Matrix Operations: MatrixOps](#abstract-matrix-operations-matrixops)


## For Loops and Iterators

Before considering matrices, let's look at some 
basic computations with vectors (in the sense of linear algebra).
Suppose you want to multiply each element of a vector by some constant.
The obvious way (if you come from C/C++) goes like this:
```rust
    let s = 1.1;
    let b = &mut[ 1.0, 2.0, 3.0 ];
    for i in 0..b.len() {
        b[i] *= s;
    }
```
In more idiomatic Rust, the for loop is replaced by
```rust
    b.iter_mut().for_each( |bi| *bi *= s );
```
In this simple example, the compiler probably produces essentially the same code 
for both versions. However, in less trivial examples, the compiler
may be able to produce more efficient code from the iterator version.

As a second example, 
suppose you want to add a scaled version of a vector to some other vector,
like this:
```rust
    let s = 1.1;
    const N: usize = 3;
    let b: &[f64; N] = &[ 1.0, 2.0, 3.0 ];
    let c = &mut[0.0; N];
    for i in 0..N {
        c[i] = s * b[i];
    }
```
The for loop in this example can be replaced by
```rust
    b.iter().zip(c.iter_mut()).for_each( |(bi,ci)| *ci = s * bi );
```

As a final example, the following function computes 
the dot product of two vectors: 
```rust
fn dotproduct(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "mismatched lengths");
    a.iter().zip(b.iter()).fold(0.0, |sum, (ai, bi)| sum + ai*bi)
}
```
However, it may be preferable to implement this function 
for all numeric types simultaneously, like this:
```rust
fn dotproduct <T: Scalar> (a: &[T], b: &[T]) -> T
{
    assert_eq!(a.len(), b.len(), "mismatched lengths");
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}
```
In this case, getting the iterator version right is challenging.


## BLAS and OxiBLAS

Matrix-times-vector and matrix-times-matrix multiplications
can, of course, be written with for loops or with iterators as above.
However, even the best compilers (for any programming language) 
cannot by themselves translate such naive implementations
into efficient code for modern hardware. 

The standard way to deal with this situation are 
[BLAS libraries](https://www.netlib.org/blas),
which provide a standard set of basic linear-algebra computations
with implementations optimized for different hardware platforms.
The two most important BLAS routines are 
`gemv()` for matrix-times-vector multiplication
and `gemm()` for matrix-times-matrix multiplication.

BLAS libraries have traditionally been written in FORTRAN or C.
Until recently, using BLAS libraries from Rust 
required the installation and configuration of such external BLAS libraries,
which can be a headache.

Fortunately, using BLAS from Rust has become much easier 
with [OxiBLAS](https://lib.rs/crates/oxiblas),
which is a BLAS implementation that is written in Rust 
and adapts itself to the hardware.

The pivotal data types to work with matrices 
in OxiBLAS are `MatRef` and `MatMut`,
which are just views; they don't own the actual matrix (i.e., its entries),
but are rich pointers to it. 
`MatRef` is a read-only view while `MatMut` allows to write into the matrix.
`MatRef` is a struct containing
- a pointer (reference) to the matrix entry at [0,0]
- the number of rows
- the number of columns
- the row stride (= the stride along rows)
- the column stride (= the stride along columns)
- a lifetime marker (which the user need not care about)

The ability of `MatRef` to work with general strides for both rows and columns is crucial
for the flexibility and efficiency of OxiBLAS.
`MatMut` is essentially the same struct as `MatRef`, 
except that the column stride of `MatMut` is fixed to 1 
(in agreement with the traditional FORTRAN convention). 

For example, the BLAS function `gemm` computes `matrix_c = alpha * matrix_a * matrix_b + beta * matrix_c`.
In OxiBLAS, `gemm()` is declared roughly as follows:

```rust
pub fn gemm<T>(
    alpha: T,
    matrix_a: MatRef<T>,
    matrix_b: MatRef<T>,
    beta: T,
    matrix_c: MatMut<T>
)
```

For storing the actual matrix entries, OxiBLAS provides the type `Mat`,
which stores the entries in the FORTRAN convention 
(i.e., column by column, with column stride = 1).

Example 1:
```rust
use oxiblas::prelude::*;

let a = Mat::from_rows(&[
	&[1.0, 2.0, 3.0],
	&[10.0, 20.0, 30.0],
	]);
assert_eq!( a[(1,0)], 10.0 );

let a_view: MatRef<f64> = a.as_ref();
assert_eq!( a_view[(1,0)], 10.0 );
```

Example 2:
```rust
use oxiblas::prelude::*;

let mut b: Mat<f64> = Mat::zeros(3, 2);
let mut b_view: MatMut<f64> = b.as_mut();
b_view[(2,1)] = 2.0;
assert_eq!( b[(2,1)], 2.0 );
```

Example 3: calling an OxiBLAS function:
```rust
use oxiblas::prelude::*;

let a = Mat::from_rows(&[
	&[1.0, 2.0, 3.0],
	&[10.0, 20.0, 30.0],
	]);
let b = Mat::from_rows(&[
	&[0.1, 0.01],
	&[0.2, 0.02],
	&[0.3, 0.03],
	]);
let mut c: Mat<f64> = Mat::zeros(2, 2);
gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
```

However, matrices need not be stored as `Mat`. For example:
```rust
use oxiblas::prelude::*;

const A_NROWS: usize = 2;
const A_NCOLS: usize = 3;
let a: &[f64; A_NROWS*A_NCOLS] = &[
	1.0, 2.0, 3.0,
	10.0, 20.0, 30.0
	];
let a_view = MatRef::from_strided(a, A_NROWS, A_NCOLS, 1, A_NCOLS).unwrap();

const B_NROWS: usize = 3;
const B_NCOLS: usize = 2;
let b: &[f64; B_NROWS*B_NCOLS] = &[
	0.1, 0.01,
	0.2, 0.02,
	0.3, 0.03,
	];
let b_view = MatRef::from_strided(b, B_NROWS, B_NCOLS, 1, B_NCOLS).unwrap();

const C_NROWS: usize = 2;
const C_NCOLS: usize = 2;
let c_tr = &mut[0.0; C_NROWS*C_NCOLS];
let mut c_view = MatMut::from_strided(c_tr, C_NROWS, C_NCOLS, C_NCOLS).unwrap();

gemm(1.0, a_view, b_view, 0.0, c_view.rb_mut());
assert_eq!( c_view[(1,0)], 14.0 );
```
Reborrowing with 
`rb_mut()` is required here to satisfy the borrow checker.


## Abstract Matrix Operations: MatrixOps

So far, this has all been standard. 
By contrast, MatrixOps is something I have been writing myself 
in order to address a problem that frequently occurs in my work,
viz., to implement algorithms so that they work efficiently
with different matrices or linear operators 
that may have specialized low-complexity implementations. 
To this end, MatrixOps defines the traits `MatrixOps1` and `MatrixOps2`
for matrix-times-vector multiplications and matrix-times-matrix multiplications,
respecively. 
For example,
`MatrixOps1` contains the method 
```rust
// vct_c =  self * vct_b
fn mult_vct(&self, vct_b: &[Float], vct_c: &mut [Float]);
 ```
and `MatrixOps2` contains the method 
```rust
// mat_c = alpha * self * mat_b + beta * mat_c
fn mult_mat_scaled_addto(&self, alpha: Float, mat_b: &MatRef<Float>, beta: Float, mat_c: &mut MatMut<Float>);
 ```
which is an abstract version of the BLAS function `gemm`.
Note that only the matrix `self` is abstract while the operands `vct_b` and `vct_c`
are slices and the operands `mat_b` and `mat_c` are OxiBLAS views.
MatrixOps also provides an implementation of these traits 
for generic matrices using OxiBLAS.

The traits in MatrixOps should be slim because
using an algorithm that is written with these abstractions
requires the user to implement `MatrixOps1` or `MatrixOps2` or both 
for the specific matrices of the application.

