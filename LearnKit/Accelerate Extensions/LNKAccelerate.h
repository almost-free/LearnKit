//
//  LNKAccelerate.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Accelerate/Accelerate.h>

#define UNIT_STRIDE 1

#if USE_DOUBLE_PRECISION

#define LNK_vfill		vDSP_vfillD
#define LNK_vclr		vDSP_vclrD
#define LNK_vmean		vDSP_meanvD
#define LNK_vsadd		vDSP_vsaddD
#define LNK_dotpr		vDSP_dotprD
#define LNK_vadd		vDSP_vaddD
#define LNK_vdiv		vDSP_vdivD
#define LNK_vsub		vDSP_vsubD
#define LNK_vsdiv		vDSP_vsdivD
#define LNK_mmul		vDSP_mmulD
#define LNK_mmov		vDSP_mmovD
#define LNK_vmul		vDSP_vmulD
#define LNK_vsmul		vDSP_vsmulD
#define LNK_mtrans		vDSP_mtransD
#define LNK_vneg		vDSP_vnegD
#define LNK_svdiv		vDSP_svdivD
#define LNK_vsum		vDSP_sveD
#define LNK_vsq			vDSP_vsqD
#define LNK_vlog		vvlog
#define LNK_vexp		vvexp

#define LNK_sqrt		sqrt
#define LNK_pow			pow
#define LNK_exp			exp
#define LNK_strtoflt(str,len)	strtod((str), (char **)((str) + (len)))

/// Computes the euclidean distance between the two vectors.
#define LNKVectorDistance(vector1, vector2, outDistance, n) vDSP_distancesqD((vector1), UNIT_STRIDE, (vector2), UNIT_STRIDE, (outDistance), (n))

#else

#define LNK_vfill		vDSP_vfill
#define LNK_vclr		vDSP_vclr
#define LNK_vmean		vDSP_meanv
#define LNK_vsadd		vDSP_vsadd
#define LNK_dotpr		vDSP_dotpr
#define LNK_vadd		vDSP_vadd
#define LNK_vdiv		vDSP_vdiv
#define LNK_vsub		vDSP_vsub
#define LNK_vsdiv		vDSP_vsdiv
#define LNK_mmul		vDSP_mmul
#define LNK_mmov		vDSP_mmov
#define LNK_vmul		vDSP_vmul
#define LNK_vsmul		vDSP_vsmul
#define LNK_mtrans		vDSP_mtrans
#define LNK_vneg		vDSP_vneg
#define LNK_svdiv		vDSP_svdiv
#define LNK_vsum		vDSP_sve
#define LNK_vsq			vDSP_vsq
#define LNK_vlog		vvlogf
#define LNK_vexp		vvexpf

#define LNK_sqrt		sqrtf
#define LNK_pow			powf
#define LNK_exp			expf
#define LNK_strtoflt(str,len)	strtof((str), (char **)((str) + (len)))

/// Computes the euclidean distance between the two vectors.
#define LNKVectorDistance(vector1, vector2, outDistance, n) vDSP_distancesq((vector1), UNIT_STRIDE, (vector2), UNIT_STRIDE, (outDistance), (n))

#endif

/* In-place operations */

/// Inverts a matrix of dimensions n * n.
extern void LNK_minvert(LNKFloat *matrix, LNKSize n);

/// Applies the sigmoid function to every element of the vector.
extern void LNK_vsigmoid(LNKFloat *vector, LNKSize n);

/* Out-of-place operations */

/// Applies the gradient of the sigmoid function to every element of the sigmoid vector.
extern void LNK_vsigmoidgrad(const LNKFloat *vector, LNKFloat *outVector, LNKSize n);

/// Computes the standard deviation of the elements in the vector.
extern LNKFloat LNK_vsd(const LNKFloat *vector, LNKSize n, LNKSize stride, LNKFloat *workgroup, LNKFloat mean, BOOL inSample);

/// Computes the determinant of the n * n matrix.
extern LNKFloat LNK_mdet(const LNKFloat *matrix, LNKSize n);
