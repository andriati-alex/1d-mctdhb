#ifndef _array_h
#define _array_h

#include <complex.h>
#include <mkl.h>





/* DATATYPES OF THE PACKAGE
 * *********************************************************
 *
 * Description
 * ===========
 *
 * Header with basic datatypes used in this project
 *
 *
 *
 * The Compressed-Column-Storage(CCS) of a Matrix
 * ==============================================
 *
 * This method consist in save memory when  storing a
 * sparse Matrix by shifting all the nonzero elements
 * to the left. Then the resulting matrix has only  m
 * columns,  m being  the maximum  number of  nonzero
 * elements in a same row, and the same number of row
 * than the former one.  Thereafter  this  matrix  is
 * organized in a  vector  following  a  Column Major
 * ordering as well as its former column positions are
 * stored in a vector of integers of the same size.
 *
 * ********************************************************/





/************************ Constants ***********************/

#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102
#define PI 3.141592653589793





/***************** Shortcut double complex ****************/

typedef double complex doublec;





/********************** MKL Pointers **********************/

typedef MKL_Complex16 * CMKLarray;





/********************** Integer vectors **********************/

typedef int * Iarray;





/***************** Real vector and matrix *****************/

typedef double *  Rarray;
typedef double ** Rmatrix;





/**************** Complex vector and matrix ***************/

typedef double complex *  Carray;
typedef double complex ** Cmatrix;





/******* Compressed Column storage of an n x n Matrix *****/

struct CCS{
    int  m;     // max number of non-zero elements in a same row
	int * col;  // Column index of elemetns.
	Carray vec; // column oriented vector.
};

typedef struct CCS * CCSmat;





/**** Compressed Column storage of an n x n Real Matrix ****/

struct RCCS{
    int  m;     // max number of non zero elements in a same row
	int * col;  // Column index of elemetns.
	Rarray vec; // column oriented vector.
};

typedef struct RCCS * RCCSmat;

#endif
