#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#define NR_CLASS 11	/* number of classes */
#define NR_FEATURE 10 /* number of features */
#define NR_L 349	/* total #SV */
#define NR_PAIR NR_CLASS * (NR_CLASS - 1) / 2
#define FORMAT "%lf\n"
#define PREC "%lf\n"

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

typedef double real_t;
typedef short int_t;

typedef struct svm_node {
	int_t index;
	real_t value;
} Node;

typedef struct Sample {
    Node data[NR_FEATURE];
} Sample;

typedef struct svm_model {
	int_t svm_type;
	int_t kernel_type;
	real_t gamma;
	Node SV[NR_L][NR_FEATURE];		/* SVs (SV[l]) */
	real_t sv_coef[NR_FEATURE][NR_L];	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	real_t rho[NR_PAIR];		/* constants in decision functions (rho[k*(k-1)/2]) */
	int_t label[NR_CLASS];		/* label of each class (label[k]) */
	int_t nSV[NR_CLASS];		/* number of SVs for each class (nSV[k]) */ 
} SVM;

static SVM model;
static Sample test_sample;
//svm-train -s 0 -c 5 -t 2 -g 0.8 -e 0.5 data/vowel.scale

static real_t max = -1e6, min = 1e6;

real_t round_real(real_t x) {
	if (x > max) {
		max = x;
	}
	if (x < min) {
		min = x;
	}
	real_t result = 0;
	char buffer[128];
	memset(buffer, 0, sizeof(buffer));
	sprintf(buffer, PREC, x);
	sscanf(buffer, FORMAT, &result);
	return result;
}

void svm_load_model(const char *model_file_name)
{
	memset(&model, 0, sizeof(model));
	model.svm_type = NU_SVC;
	model.kernel_type = RBF;
	model.gamma = 0.8;

	FILE *fp = fopen(model_file_name,"rb");

	char buffer[128];
	int i = 0, j = 0;

	// get rhos
	fscanf(fp, "%s", buffer);
	for (i = 0; i < NR_PAIR; i++) {
		fscanf(fp, FORMAT, &model.rho[i]);
		model.rho[i] = round_real(model.rho[i]);
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
			fscanf(fp, FORMAT, &model.sv_coef[i][j]);
			model.sv_coef[i][j] = round_real(model.sv_coef[i][j]);
		}
		// get SV
		for (i = 0; i < NR_FEATURE; i++) {
			fscanf(fp, FORMAT, &model.SV[j][i].value);
			model.SV[j][i].index = (i + 1);
			model.SV[j][i].value = round_real(model.SV[j][i].value);
		}
	}

	fclose(fp);
}

void print_model(const char *model_file_name) {
	FILE *output1 = fopen (model_file_name, "w");

	int i = 0, j = 0;
	fprintf(output1, "rho\n");
	for (j = 0; j < 55; j++) {
		fprintf(output1, "%lf ", model.rho[j]);
	}
	fprintf(output1, "\n");
	fprintf(output1, "label\n");
	for (j = 0; j < 11; j++) {
		fprintf(output1, "%d ", model.label[j]);
	}
	fprintf(output1, "\n");
	fprintf(output1, "nSV\n");
	for (j = 0; j < 11; j++) {
		fprintf(output1, "%d ", model.nSV[j]);
	}
	fprintf(output1, "\n");
	for (j = 0; j < NR_L; j++) {
		fprintf(output1, "SV-No.%d\n", (j + 1));
		for (i = 0; i < 10; i++) {
			fprintf(output1, "%.15f\n", model.sv_coef[i][j]);
		}
		for (i = 0; i < 10; i++) {
			fprintf(output1, "%.7f\n", model.SV[j][i].value);
		}
	}	
	fclose(output1);
}

real_t rbf_kernel(const Node x[NR_FEATURE], const Node y[NR_FEATURE]) {
	real_t sum = 0;
	int i = 0;
	for (i = 0; i < NR_FEATURE; i++) {
		//printf("x=%d y=%d\n", x[i].index, y[i].index);
		real_t d = x[i].value - y[i].value;
		sum += d*d;
	}
	return exp(-model.gamma * sum);
}

static real_t dec_values[NR_PAIR];
static real_t kvalue[NR_L];
static int_t start[NR_CLASS];
static int_t vote[NR_CLASS];

int_t svm_predict(const Sample sample) {
	//const Node *x = sample.data;
	int_t i;

	for(i = 0; i < NR_L; i++)
		kvalue[i] = rbf_kernel(sample.data, model.SV[i]);

	start[0] = 0;
	for(i = 1; i < NR_CLASS; i++)
		start[i] = start[i - 1] + model.nSV[i - 1];

	for(i = 0; i < NR_CLASS; i++)
		vote[i] = 0;

	int_t p=0;
	for(i=0;i<NR_CLASS;i++){
		for(int j=i+1;j<NR_CLASS;j++) {
			real_t sum = 0;
			int_t si = start[i];
			int_t sj = start[j];
			int_t ci = model.nSV[i];
			int_t cj = model.nSV[j];
			
			int_t k;
			real_t *coef1 = model.sv_coef[j-1];
			real_t *coef2 = model.sv_coef[i];
			for(k=0;k<ci;k++)
				sum += coef1[si+k] * kvalue[si+k];
			for(k=0;k<cj;k++)
				sum += coef2[sj+k] * kvalue[sj+k];
			sum -= model.rho[p];
			dec_values[p] = sum;

			if(dec_values[p] > 0)
				++vote[i];
			else
				++vote[j];
			p++;
		}
	}

	int vote_max_idx = 0;
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
	for (i = 0; i < n; i++) {
		int_t label = -1;
		fscanf(input, "%hd", &label);
		for (j = 0; j < NR_FEATURE; j++) {
			fscanf(input, FORMAT, &test_sample.data[j].value);
			test_sample.data[j].value = round_real(test_sample.data[j].value);
			test_sample.data[j].index = (j + 1);
		}
		int_t predict = svm_predict(test_sample);
		//printf("%d\n", predict);
		if (predict == label) {
			correct++;
		}
	}
	fclose(input);
	printf("%d %d\n", correct, n);
	return (real_t)correct / (real_t)n;
}

int main () {
	svm_load_model("model_reduced.txt");
	//print_model("model_get.txt");
	printf(FORMAT, predict_sample(("testcase_200.txt")));
	printf(FORMAT, round_real(2.38123453));
	printf(FORMAT, min);
	printf(FORMAT, max);
	return 0;
}
