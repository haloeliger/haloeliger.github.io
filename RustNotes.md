# On Elementary Matrix Computations in Rust #

*Version 1.1, Oct. 16, 2026*

I am a big fan of Rust. 
However, like many others, I found that learning Rust 
is not so easy.

In this note, I summarize what I learned about
matrix-times-vector and matrix-times-matrix multiplications,
which are the computational backbone of most scientific computing.

Many thanks go to Hampus Malmberg, Hugo Aguettaz, and Cyrill Achermann,
who helped me a lot to get started with Rust.


- [For Loops and Iterators](#for-loops-and-iterators)
- [Generics](#generics)
- [BLAS and OxiBLAS](#blas-and-oxiblas)
- [Abstract Matrix Operations: MatrixOps](#abstract-matrix-operations-matrixops)


## For Loops and Iterators

Before considering matrices, let's look at some 
basic computations with vectors (in the sense of linear algebra).
Suppose you want to multiply each element of a vector by some scale factor.
The obvious way (if you come from C/C++) goes like this:
```ignore
// the vector, e.g.:
let b = &mut[ 1.0, 2.0, 3.0 ];
// the scale factor, e.g.:
let s = 1.1;
// the multiplication:
for i in 0..b.len() {
	b[i] *= s;
}
```
In more idiomatic Rust, the **for** loop is replaced by
```ignore
    b.iter_mut().for_each( |pbi| *pbi *= s );
```

As a second example, 
suppose you want to add a scaled version of a vector to some other vector,
like this:
```ignore
const N: usize = 3;
let b: &[f64; N] = &[ 1.0, 2.0, 3.0 ];
let s = 1.1;
let c = &mut[0.0; N];
for i in 0..N {
	c[i] = s * b[i];
}
```
The **for** loop in this example can be replaced by
```ignore
    b.iter().zip(c.iter_mut()).for_each( |(pbi, pci)| *pci = s * *pbi );
```

## Generics

The following function computes the dot product of two vectors 
for all numeric types simultaneously:
```ignore
fn dotproduct <T: Scalar> (a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len(), "mismatched lengths");
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}
```
Or more compactly like this:
```ignore
fn gendotproduct2 <T: Scalar> (a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len(), "mismatched lengths");
    a.iter().zip(b.iter()).fold( T::zero(), |sum, (pai, pbi)| sum + *pai * *pbi )
}
```



## BLAS and OxiBLAS

Matrix-times-vector and matrix-times-matrix multiplications
can, of course, be written with **for** loops or with iterators as above.
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
but it allows writing into the matrix and 
the column stride is fixed to 1 
(in agreement with the traditional FORTRAN convention). 

For example, the BLAS function `gemm` computes `matrix_c = alpha * matrix_a * matrix_b + beta * matrix_c`.
In OxiBLAS, `gemm` is declared roughly as follows:

```ignore
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
(i.e., column by column, with column stride 1).

Example with Mat and MatRef:
```ignore
use oxiblas::prelude::*;

let a = Mat::from_rows(&[
	&[1.0, 2.0, 3.0],
	&[10.0, 20.0, 30.0],
	]);
assert_eq!( a[(1,0)], 10.0 );

let a_view: MatRef<f64> = a.as_ref();
assert_eq!( a_view[(1,0)], 10.0 );
```

Example with Mat and MatMut:
```ignore
use oxiblas::prelude::*;

let mut b: Mat<f64> = Mat::zeros(3, 2);
let mut b_view: MatMut<f64> = b.as_mut();
b_view[(2,1)] = 2.0;
assert_eq!( b[(2,1)], 2.0 );
```

Example of calling `gemm`:
```ignore
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
assert_eq!( c[(1,0)], 14.0 );
```

However, matrices need not be stored as `Mat`; slices will do.
(And remember that MatMut assumes the FORTRAN convention).
For example:
```ignore
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
assert_eq!( c_tr[1], 14.0 );
```
In this example, reborrowing `c_view` with 
`rb_mut()` is required in the call of `gemm` to satisfy the borrow checker.


## Abstract Matrix Operations: MatrixOps

All of the above is standard.
By contrast, [MatrixOps](https://github.com/haloeliger/MatrixOps) is something I wrote myself 
in order to address a problem that frequently occurs in my work,
viz., to implement algorithms so that they 
work efficiently not only with generic matrices, but 
also with linear operators 
that have specialized low-complexity implementations. 
To this end, MatrixOps defines the traits `MatrixOps1` and `MatrixOps2`
for matrix-times-vector multiplications and matrix-times-matrix multiplications,
respectively.
For example,
`MatrixOps1` defines the method 
```ignore
// vct_c =  self * vct_b
fn mult_vct(&self, vct_b: &[Float], vct_c: &mut[Float]);
 ```
(where `Float` is a generic numerical scalar), 
and `MatrixOps2` defines the method 
```ignore
// mat_c = alpha * self * mat_b + beta * mat_c
fn mult_mat_scaled_addto(&self, alpha: Float, mat_b: &MatRef<Float>, beta: Float, mat_c: &mut MatMut<Float>);
 ```
which is an abstract version of the BLAS function `gemm`.
Note that only the operator `self` is abstract while the operands 
and the results are slices (in `MatrixOps1`) or OxiBLAS views (in `MatrixOps2`).
MatrixOps also provides an implementation of these traits 
for generic matrices using OxiBLAS. 
See the documentation of [MatrixOps](https://github.com/haloeliger/MatrixOps) for examples.
