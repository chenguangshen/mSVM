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
#define FORMAT "%f\n"
#define PREC "%f\n"
#define SCALE 1000.0

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

typedef short data_type;
typedef float real_t;
typedef short int_t;

typedef struct svm_node {
	int_t index;
	data_type value;
} Node;

typedef struct Sample {
    Node data[NR_FEATURE];
} Sample;

typedef struct svm_model {
	int_t svm_type;
	int_t kernel_type;
	real_t gamma;
	Node SV[NR_L][NR_FEATURE];		/* SVs (SV[l]) */
	data_type sv_coef[NR_FEATURE][NR_L];	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	data_type rho[NR_PAIR];		/* constants in decision functions (rho[k*(k-1)/2]) */
	int_t label[NR_CLASS];		/* label of each class (label[k]) */
	int_t nSV[NR_CLASS];		/* number of SVs for each class (nSV[k]) */ 
} SVM;

static SVM model;
static Sample test_sample;
//svm-train -s 0 -c 5 -t 2 -g 0.8 -e 0.5 data/vowel.scale

static real_t max = -1e6, min = 1e6;

data_type round_real(real_t x) {

	if (x > max) {
		max = x;
	}
	if (x < min) {
		min = x;
	}
	//data_type i = round(x * SCALE);
	//printf("%f %d\n", x, i);
//	printf("x=%f\n", x);
//	data_type res = SCALE * x;
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
	int_t i = 0, j = 0;
	real_t temp = 0;

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

// 	int_t i = 0, j = 0;
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
// we try to minimize the number of fp operations
data_type rbf_kernel(const Node x[NR_FEATURE], const Node y[NR_FEATURE]) {
	int sum = 0;
	int_t i = 0;
	for (i = 0; i < NR_FEATURE; i++) {
		//printf("x=%d y=%d\n", x[i].value, y[i].value);
		int d = x[i].value - y[i].value;
		//printf("%f\n", d);
		sum += d*d;
	}
	//#############################################
	// this is the only two lines of floating point operation
	real_t res = exp(-model.gamma * (real_t)sum / SCALE / SCALE);
	data_type res_scale = res * SCALE;
	//#############################################
	// printf("res=%f\n", res);
	// printf("scaled res=%d\n", res_scale);
	return res_scale;
}

static data_type dec_values[NR_PAIR];
static data_type kvalue[NR_L];
static data_type start[NR_CLASS];
static data_type vote[NR_CLASS];

data_type svm_predict(const Sample sample) {
	//const Node *x = sample.data;
	data_type i;

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

	data_type p=0, j = 0;
	for(i=0;i<NR_CLASS;i++){
		for(j=i+1;j<NR_CLASS;j++) {
			data_type sum = 0;
			data_type si = start[i];
			data_type sj = start[j];
			data_type ci = model.nSV[i];
			data_type cj = model.nSV[j];
			
			data_type k;
			data_type *coef1 = model.sv_coef[j-1];
			data_type *coef2 = model.sv_coef[i];
			for(k=0;k<ci;k++){
				//printf("%d %f\n", coef1[si+k], kvalue[si+k]);
				//printf("%d %f\n", a, b);
				//data_type ttt = coef1[si+k] * kvalue[si+k];
				//printf("%d\n", ttt);
				sum += coef1[si+k] * kvalue[si+k] / 1000;
			}
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

	int_t vote_max_idx = 0;
	for(i=1; i < NR_CLASS; i++)
		if(vote[i] > vote[vote_max_idx])
			vote_max_idx = i;

	return model.label[vote_max_idx];
}

real_t predict_sample(const char *test_sample_name) {
	int_t correct = 0;
	FILE *input = fopen(test_sample_name, "r");
	int_t i = 0, j = 0;
	int_t n = -1;
	fscanf(input, "%hd", &n);
	real_t temp;
	for (i = 0; i < n; i++) {
		int_t label = -1;
		fscanf(input, "%hd", &label);
		for (j = 0; j < NR_FEATURE; j++) {
			fscanf(input, FORMAT, &temp);
			//printf("%lf\n", temp);
			test_sample.data[j].value = round_real(temp);
			test_sample.data[j].index = (j + 1);
		}
		int_t predict = svm_predict(test_sample);
		//printf("%d\n", predict);
		if (predict == label) {
			correct++;
		}
	}
	fclose(input);
	printf("%hd %hd\n", correct, n);
	return (real_t)correct / (real_t)n;
}

int_t main () {
	svm_load_model("model/model_reduced.txt");
	//print_model("model_get.txt");
	hexagon_sim_init_timer();
	hexagon_sim_start_timer();
	real_t result = predict_sample("data/testcase_200.txt");
	printf("%lf\n", result);
	hexagon_sim_end_timer();
	hexagon_sim_show_timer(stdout);
	printf("test scaling\n");

	real_t xxx = 0.5;
	data_type a = round_real(xxx);
	printf("%d\n", a);
	
	printf(FORMAT, min);
	printf(FORMAT, max);
	return 0;
}

// Currently, this implementation uses 64666912 cycles - 0.6ms for each sample
