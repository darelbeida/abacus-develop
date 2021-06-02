#ifndef INVERSE_MATRIX_PERTURBE_UNITTEST_H
#define INVERSE_MATRIX_PERTURBE_UNITTEST_H

#include "../../src_tools/inverse_matrix_perturbe.h"
#include "common_test.h"

void Inverse_Matrix_Perturbe_Unittest()
{
	const size_t dim(2);
	
	ComplexMatrix A(dim,dim), B(dim,dim), A_i(dim,dim);
	A(0,0)=1; A(0,1)=3; A(1,0)=5; A(1,1)=6;
	B(0,0)=0.2; B(0,1)=0.1; B(1,0)=0.4; B(1,1)=0.3;
	A_i(0,0)=-0.66666667; A_i(0,1)=0.33333333; A_i(1,0)=0.55555556; A_i(1,1)=-0.11111111;

	Inverse_Matrix_Perturbe inverse;	
	inverse.init(dim);
	inverse.set_A_inverse(&A_i).set_B(&B);
	
	for( int plus_minus_state(1); plus_minus_state<=2; ++plus_minus_state)
	{
		inverse.set_plus_minus_state(plus_minus_state);
		cout<<"====="<<plus_minus_state<<"====="<<endl;
		for(size_t iterate_num(0); iterate_num<10; ++iterate_num)
		{
			cout<<iterate_num<<endl;
			const ComplexMatrix &ii(inverse.set_iterate_num(iterate_num).cal_inverse());
			cout_matrix(&ii,dim);
		}
	}

}

#endif

/* output:
=====1=====
0
(-0.666667,0)	(0.333333,0)	
(0.555556,0)	(-0.111111,0)	

1
(-0.685185,0)	(0.337037,0)	
(0.587654,0)	(-0.130864,0)	

2
(-0.686255,0)	(0.337695,0)	
(0.588176,0)	(-0.130672,0)	

3
(-0.686273,0)	(0.337689,0)	
(0.588235,0)	(-0.13072,0)	

4
(-0.686275,0)	(0.337691,0)	
(0.588235,0)	(-0.130719,0)	

5
(-0.686275,0)	(0.337691,0)	
(0.588235,0)	(-0.130719,0)	

6
(-0.686275,0)	(0.337691,0)	
(0.588235,0)	(-0.130719,0)	

7
(-0.686275,0)	(0.337691,0)	
(0.588235,0)	(-0.130719,0)	

8
(-0.686275,0)	(0.337691,0)	
(0.588235,0)	(-0.130719,0)	

9
(-0.686275,0)	(0.337691,0)	
(0.588235,0)	(-0.130719,0)	

=====2=====
0
(-0.666667,0)	(0.333333,0)	
(0.555556,0)	(-0.111111,0)	

1
(-0.648148,0)	(0.32963,0)	
(0.523457,0)	(-0.091358,0)	

2
(-0.649218,0)	(0.330288,0)	
(0.523978,0)	(-0.091166,0)	

3
(-0.649201,0)	(0.330294,0)	
(0.523918,0)	(-0.0911178,0)	

4
(-0.649203,0)	(0.330296,0)	
(0.523918,0)	(-0.0911163,0)	

5
(-0.649203,0)	(0.330296,0)	
(0.523918,0)	(-0.0911162,0)	

6
(-0.649203,0)	(0.330296,0)	
(0.523918,0)	(-0.0911162,0)	

7
(-0.649203,0)	(0.330296,0)	
(0.523918,0)	(-0.0911162,0)	

8
(-0.649203,0)	(0.330296,0)	
(0.523918,0)	(-0.0911162,0)	

9
(-0.649203,0)	(0.330296,0)	
(0.523918,0)	(-0.0911162,0)	
*/