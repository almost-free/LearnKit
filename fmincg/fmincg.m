/*
Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
%
%
% (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
% 
% Permission is granted for anyone to copy, use, or modify these
% programs and accompanying documents for purposes of research or
% education, provided this copyright notice is retained, and note is
% made of any changes that have been made.
% 
% These programs and documents are distributed without any warranty,
% express or implied.  As the programs were written for research
% purposes only, they have not been tested to the degree that would be
% advisable in any important application.  All use of these programs is
% entirely at the user's own risk.

[ Sagar GV, 2013, sagar.writeme@gmail.com ] Changes Made:
- Ported to C

*/
#include "fmincg.h"

int fmincg(void (*costFunc)(LNKFloat *inputVector, LNKFloat *cost, LNKFloat *gradVector), LNKFloat *xVector, int nDim, int maxCostCalls)
{
	int success = 0,costFuncCount=0,lineSearchFuncCount=0;
	LNKFloat ls_failed,f1,d1,z1,f0,f2,d2,f3,d3,z3,limit,z2,A,B,C;
	LNKFloat df1[nDim],s[nDim],x0[nDim],df0[nDim],df2[nDim],df2neg[nDim],tmp[nDim];
	LNKFloat *x = xVector;

	ls_failed = 0;
	
	if (costFuncCount >= maxCostCalls) {
		return 1;
	} else {
		costFuncCount++;
	}

	(*costFunc)(xVector,&f1,df1);

	LNK_vneg(df1, UNIT_STRIDE, s, UNIT_STRIDE, nDim);

	LNK_dotpr(s, UNIT_STRIDE, s, UNIT_STRIDE, &d1, nDim);
	d1 = -d1;

	z1 = 1.0f / (1 - d1);
	
	while (1) {
		memcpy(x0, x, sizeof(LNKFloat) * nDim);
		memcpy(df0, df1, sizeof(LNKFloat) * nDim);

		f0 = f1;

		LNK_vsma(s, UNIT_STRIDE, &z1, x, UNIT_STRIDE, x, UNIT_STRIDE, nDim);
		
		if (costFuncCount >= maxCostCalls) {
			return 1;
		} else {
			costFuncCount++;
		}

		(*costFunc)(x,&f2,df2);

		LNK_dotpr(df2, UNIT_STRIDE, s, UNIT_STRIDE, &d2, nDim);

		f3 = f1;
		d3 = d1;
		z3 = -z1;
		
		success = 0; 
		limit = -1;
		lineSearchFuncCount = 0;

		// begin line search
		while (1) {
			while ((( (f2) > ((f1) + RHO*(z1)*(d1))) || ( (d2) > -SIG*(d1) )) && lineSearchFuncCount < MAXV) {
				limit = z1;

				if ( (f2) > (f1) ) {
					z2 = z3 - (0.5f*(d3)*(z3)*(z3))/((d3)*(z3)+(f2)-(f3));
				}
				else {
					A = 6*((f2)-(f3))/(z3)+3*((d2)+(d3));
					B = 3*((f3)-(f2))-(z3)*((d3)+2*(d2));
					z2 = (sqrt(B*B-A*(d2)*(z3)*(z3))-B)/A;
				}

				if (isnan(z2) || isinf(z2)) {
					z2 = (z3) * 0.5f;
				}

				A = ((z2 < INT*(z3)) ? z2 : INT*(z3));
				B = (1-INT)*(z3);
				z2 = A > B ? A : B;
				z1 = z1 + z2;

				LNK_vsma(s, UNIT_STRIDE, &z2, x, UNIT_STRIDE, x, UNIT_STRIDE, nDim);

				if (costFuncCount >= maxCostCalls) {
					return 1;
				} else {
					costFuncCount++;
				}

				lineSearchFuncCount++;
				(*costFunc)(x,&f2,df2);

				LNK_dotpr(df2, UNIT_STRIDE, s, UNIT_STRIDE, &d2, nDim);

				z3 = z3 - z2;
			}
			if ( (f2 > f1 + (z1)*RHO*(d1)) || ((d2) > -SIG*(d1)) ) {
				break; //failure
			}
			else if ( d2 > SIG*(d1) ) {
				success = 1; break; 
			}
			else if (lineSearchFuncCount >= MAXV) {
				break;
			}

			A = 6*(f2-f3)/z3+3*(d2+d3);
			B = 3*(f3-f2)-z3*(d3+2*d2);
			z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3));

			if (!(B*B-A*d2*z3*z3 >= 0) || isnan(z2) || isinf(z2) || z2 < 0) {
				if (limit < -0.5f) {
					z2 = z1 * (EXT-1);
				}
				else {
					z2 = (limit-z1)/2;
				}
			}
			else if((limit > -0.5) && (z2+z1 > limit)) {
				z2 = (limit-z1)/2; 
			}	
			else if((limit < -0.5) && (z2+z1 > z1*EXT)) {
				z2 = z1*(EXT-1.0);
			}
			else if(z2 < -z3*INT) {
				z2 = -z3*INT;
			}
			else if((limit > -0.5) && (z2 < (limit-z1)*(1.0-INT))) {
				z2 = (limit-z1)*(1.0-INT);
			}

			f3 = f2; d3 = d2; z3 = -z2;
			z1 = z1 + z2;

			LNK_vsma(s, UNIT_STRIDE, &z2, x, UNIT_STRIDE, x, UNIT_STRIDE, nDim);

			if (costFuncCount >= maxCostCalls) {
				return 1;
			} else {
				costFuncCount++;
			}

			lineSearchFuncCount++;
			(*costFunc)(x,&f2,df2);

			LNK_dotpr(df2, UNIT_STRIDE, s, UNIT_STRIDE, &d2, nDim);
		}

		// line search ended
		if (success) {
			f1 = f2;

			LNK_dotpr(df1, UNIT_STRIDE, df1, UNIT_STRIDE, &A, nDim);
			LNK_dotpr(df2, UNIT_STRIDE, df2, UNIT_STRIDE, &B, nDim);
			LNK_dotpr(df1, UNIT_STRIDE, df2, UNIT_STRIDE, &C, nDim);

			const LNKFloat coeff = (B - C) / A;
			LNK_vneg(df2, UNIT_STRIDE, df2neg, UNIT_STRIDE, nDim);
			LNK_vsma(s, UNIT_STRIDE, &coeff, df2neg, UNIT_STRIDE, s, UNIT_STRIDE, nDim);

			memcpy(tmp, df1, sizeof(LNKFloat) * nDim);
			memcpy(df1, df2, sizeof(LNKFloat) * nDim);
			memcpy(df2, tmp, sizeof(LNKFloat) * nDim);

			LNK_dotpr(s, UNIT_STRIDE, df1, UNIT_STRIDE, &d2, nDim);

			if (d2 > 0) {
				LNK_vneg(df1, UNIT_STRIDE, s, UNIT_STRIDE, nDim);

				LNK_dotpr(s, UNIT_STRIDE, s, UNIT_STRIDE, &d2, nDim);
				d2 = -d2;
			}

			A = d1 / (d2 - COST_FUNC_DATATYPE_MIN);
			z1 = z1 * ((RATIO < A) ? RATIO : A);
			d1 = d2;
			ls_failed = 0;
		}
		else {
			f1 = f0;

			memcpy(x, x0, sizeof(LNKFloat) * nDim);
			memcpy(df1, df0, sizeof(LNKFloat) * nDim);

			if (ls_failed) {
				break;
			}

			memcpy(tmp, df1, sizeof(LNKFloat) * nDim);
			memcpy(df1, df2, sizeof(LNKFloat) * nDim);
			memcpy(df2, tmp, sizeof(LNKFloat) * nDim);

			LNK_vneg(df1, UNIT_STRIDE, s, UNIT_STRIDE, nDim);

			LNK_dotpr(s, UNIT_STRIDE, s, UNIT_STRIDE, &d1, nDim);
			d1 = -d1;

			z1 = 1 / (1 - d1);
			ls_failed = 1;
		}
	}

	return 2;
}
