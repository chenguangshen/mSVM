#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "hexagon_sim_timer.h"

#define NR_CLASS 11	/* number of classes */
#define NR_FEATURE 10 /* number of features */
#define NR_L 349	/* total #SV */
#define NR_PAIR NR_CLASS * (NR_CLASS - 1) / 2
#define SCALE 1000.0

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

typedef short Word16;
typedef int Word32;
typedef float Real32;

typedef struct svm_node {
	Word16 index;
	Word16 value;
} Node;

typedef struct Sample {
    Node data[NR_FEATURE];
} Sample;

typedef struct svm_model {
	Word16 svm_type;
	Word16 kernel_type;
	Real32 gamma;
	Node SV[NR_L][NR_FEATURE];		/* SVs (SV[l]) */
	Word16 sv_coef[NR_FEATURE][NR_L];	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	Word16 rho[NR_PAIR];		/* constants in decision functions (rho[k*(k-1)/2]) */
	Word16 label[NR_CLASS];		/* label of each class (label[k]) */
	Word16 nSV[NR_CLASS];		/* number of SVs for each class (nSV[k]) */ 
} SVM;

static SVM model;
static Sample test_sample;
//svm-train -s 0 -c 5 -t 2 -g 0.8 -e 0.5 data/vowel.scale

static Real32 max = -1e6, min = 1e6;

Word16 round_real(Real32 x) {

	if (x > max) {
		max = x;
	}
	if (x < min) {
		min = x;
	}
	//Word16 i = round(x * SCALE);
	//printf("%f %d\n", x, i);
//	printf("x=%f\n", x);
//	Word16 res = SCALE * x;
//	printf("scaled x=%d\n", res);
	return SCALE * x;
}

void svm_load_model(const char *model_file_name)
{
	memset(&model, 0, sizeof(model));
	model.svm_type = NU_SVC;
	model.kernel_type = RBF;
	model.gamma = 0.8f;

	FILE *fp = fopen(model_file_name,"rb");

	char buffer[128];
	Word16 i = 0, j = 0;
	Real32 temp = 0;

	// get rhos
	fscanf(fp, "%s", buffer);
	for (i = 0; i < NR_PAIR; i++) {
		fscanf(fp, FORMAT, &temp);
		//printf("%lf\n", temp);
		model.rho[i] = round_real(temp);
	}
	
	// get labels
	fscanf(fp, "%s", buffer);
	for (i = 0; i < NR_CLASS; i++) {
		fscanf(fp, "%hd", &model.label[i]);
	}

	// get size of each SVs
	fscanf(fp, "%s", buffer);
	for (i = 0; i < NR_CLASS; i++) {
		fscanf(fp, "%hd", &model.nSV[i]);
	}

	// get sv_coef and SV
	for (j = 0; j < NR_L; j++) {
		fscanf(fp, "%s\n", buffer);
		// get sv_coef
		for (i = 0; i < NR_FEATURE; i++) {
			fscanf(fp, FORMAT, &temp);
			//printf("%lf\n", temp);
			model.sv_coef[i][j] = round_real(temp);
		}
		// get SV
		for (i = 0; i < NR_FEATURE; i++) {
			fscanf(fp, FORMAT, &temp);
			//printf("%lf\n", temp);
			model.SV[j][i].index = (i + 1);
			model.SV[j][i].value = round_real(temp);
		}
	}
	fclose(fp);
}

// void print_model(const char *model_file_name) {
// 	FILE *output1 = fopen (model_file_name, "w");

// 	Word16 i = 0, j = 0;
// 	fprintf(output1, "rho\n");
// 	for (j = 0; j < 55; j++) {
// 		fprintf(output1, "%lf ", model.rho[j]);
// 	}
// 	fprintf(output1, "\n");
// 	fprintf(output1, "label\n");
// 	for (j = 0; j < 11; j++) {
// 		fprintf(output1, "%d ", model.label[j]);
// 	}
// 	fprintf(output1, "\n");
// 	fprintf(output1, "nSV\n");
// 	for (j = 0; j < 11; j++) {
// 		fprintf(output1, "%d ", model.nSV[j]);
// 	}
// 	fprintf(output1, "\n");
// 	for (j = 0; j < NR_L; j++) {
// 		fprintf(output1, "SV-No.%d\n", (j + 1));
// 		for (i = 0; i < 10; i++) {
// 			fprintf(output1, "%.15f\n", model.sv_coef[i][j]);
// 		}
// 		for (i = 0; i < 10; i++) {
// 			fprintf(output1, "%.7f\n", model.SV[j][i].value);
// 		}
// 	}	
// 	fclose(output1);
// }


// have to use floating point for kernel computation
// otherwise will result in overflow for the exp operation

// instead of completely removing floating point,
// try to minimize the number of fp operations
Word16 rbf_kernel(const Node x[NR_FEATURE], const Node y[NR_FEATURE]) {
	Word32 sum = 0;
	Word16 i = 0;
	for (i = 0; i < NR_FEATURE; i++) {
		//printf("x=%d y=%d\n", x[i].value, y[i].value);
		Word16 d = x[i].value - y[i].value;
		//printf("%f\n", d);
		sum += d*d;
	}
	//#############################################
	// this is the only two lines of floating point operation
	Real32 res = exp(-model.gamma * (Real32)sum / SCALE / SCALE);
	Word16 res_scale = res * SCALE;
	//#############################################

	// printf("res=%f\n", res);
	// printf("scaled res=%d\n", res_scale);
	return res_scale;
}

static Word16 dec_values[NR_PAIR];
static Word16 kvalue[NR_L];
static Word16 start[NR_CLASS];
static Word16 vote[NR_CLASS];

Word16 svm_predict(const Sample sample) {
	//const Node *x = sample.data;
	Word16 i;

	for(i = 0; i < NR_L; i++) {
		kvalue[i] = rbf_kernel(sample.data, model.SV[i]);
		//printf("%f\n", kvalue[i]);
	}
	start[0] = 0;
	for(i = 1; i < NR_CLASS; i++) {
		start[i] = start[i - 1] + model.nSV[i - 1];
		//printf("%d\n", start[i]);
	}

	for(i = 0; i < NR_CLASS; i++)
		vote[i] = 0;

	Word16 p=0, j = 0;

	// 1-to-1 vote
	for(i=0;i<NR_CLASS;i++){
		for(j=i+1;j<NR_CLASS;j++) {
			Word16 sum = 0;
			Word16 si = start[i];
			Word16 sj = start[j];
			Word16 ci = model.nSV[i];
			Word16 cj = model.nSV[j];
			
			Word16 k;
			Word16 *coef1 = model.sv_coef[j-1];
			Word16 *coef2 = model.sv_coef[i];

			for(k=0;k<ci;k++)
				sum += coef1[si+k] * kvalue[si+k] / 1000;

			for(k=0;k<cj;k++)
				sum += coef2[sj+k] * kvalue[sj+k] / 1000;
			
			sum -= model.rho[p];
			//printf("%f\n", sum);
			dec_values[p] = sum;

			if(dec_values[p] > 0)
				++vote[i];
			else
				++vote[j];
			p++;
		}
	}

	Word16 vote_max_idx = 0;
	for(i=1; i < NR_CLASS; i++)
		if(vote[i] > vote[vote_max_idx])
			vote_max_idx = i;

	return model.label[vote_max_idx];
}

Real32 predict_sample(const char *test_sample_name) {
	Word16 correct = 0;
	FILE *input = fopen(test_sample_name, "r");
	Word16 i = 0, j = 0;
	Word16 n = -1;
	fscanf(input, "%hd", &n);
	Real32 temp;
	for (i = 0; i < n; i++) {
		Word16 label = -1;
		fscanf(input, "%hd", &label);
		for (j = 0; j < NR_FEATURE; j++) {
			fscanf(input, FORMAT, &temp);
			//printf("%lf\n", temp);
			test_sample.data[j].value = round_real(temp);
			test_sample.data[j].index = (j + 1);
		}
		Word16 predict = svm_predict(test_sample);
		//printf("%d\n", predict);
		if (predict == label) {
			correct++;
		}
	}
	fclose(input);
	printf("%hd %hd\n", correct, n);
	return (Real32)correct / (Real32)n;
}

Word16 main () {
	svm_load_model("model/model_reduced.txt");
	//print_model("model_get.txt");
	hexagon_sim_init_timer();
	hexagon_sim_start_timer();
	Real32 result = predict_sample("data/testcase_200.txt");
	printf("%lf\n", result);
	hexagon_sim_end_timer();
	hexagon_sim_show_timer(stdout);


	// printf("test scaling\n");
	// Real32 xxx = 0.5;
	// Word16 a = round_real(xxx);
	// printf("%d\n", a);
	
	// printf("%f\n", min);
	// printf("%f\n", max);
	return 0;
}

// Currently, this implementation uses 64666912 cycles - 0.6ms for each sample
