#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <time.h>

#define NR_SV 349
#define NR_CLASS 11	/* number of classes */
#define NR_FEATURE 10 /* number of features */
#define NR_L 349	/* total #SV */
#define NR_PAIR NR_CLASS * (NR_CLASS - 1) / 2
#define FORMAT "%f\n"
#define PREC "%f\n"

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

typedef float real_t;
typedef int int_t;

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
		//model.rho[i] = round_real(model.rho[i]);
	}
	
	// get labels
	fscanf(fp, "%s", buffer);
	for (i = 0; i < NR_CLASS; i++) {
		fscanf(fp, "%d", &model.label[i]);
	}

	// get size of each SVs
	fscanf(fp, "%s", buffer);
	for (i = 0; i < NR_CLASS; i++) {
		fscanf(fp, "%d", &model.nSV[i]);
	}

	// get sv_coef and SV
	for (j = 0; j < NR_L; j++) {
		fscanf(fp, "%s\n", buffer);
		// get sv_coef
		for (i = 0; i < NR_FEATURE; i++) {
			fscanf(fp, FORMAT, &model.sv_coef[i][j]);
			//model.sv_coef[i][j] = round_real(model.sv_coef[i][j]);
		}
		// get SV
		for (i = 0; i < NR_FEATURE; i++) {
			fscanf(fp, FORMAT, &model.SV[j][i].value);
			model.SV[j][i].index = (i + 1);
			//model.SV[j][i].value = round_real(model.SV[j][i].value);
		}
	}

	fclose(fp);
}

bool flag[NR_L];
int ind[NR_SV];
int nSV[NR_CLASS];

void setNSV(int x) {
	int cur = 0;
	int i = 0;
	while (x >= cur) {
		cur += model.nSV[i++];
	}
	nSV[--i]++;
	//printf("x=%d, i=%d\n", x, i);
}

void print_model(const char *model_file_name) {

	FILE *output1 = fopen (model_file_name, "w");
	srand(time(NULL));
	int ran;
	int count = 0;
	memset(flag, 0, sizeof(0));
	memset(nSV, 0, sizeof(nSV));

	int i = 0, j = 0;

	while (count < NR_SV) {
		ran = rand() % NR_L;
		while (flag[ran] != 0) {
			ran = rand() % NR_SV;
		}
		flag[ran] = 1;
		ind[count++] = ran;
		//printf("%d\n", ran);
	}

	printf("count=%d\n", count); 

	for (i = 0; i < NR_SV - 1; i++) {
		for (j = i; j < NR_SV; j++) {
			if (ind[i] > ind [j]) {
				int temp = ind[j];
				ind[j] = ind[i];
				ind[i] = temp;
			}
		}
	}

	for (i = 0; i < NR_SV; i++) {
	//	printf("%d\n", ind[i]);
		setNSV(ind[i]);
	}

	int cc = 0;
	for (i = 0; i < NR_CLASS; i++) {
		cc += nSV[i];
		printf("%d\n", nSV[i]);
	}
	printf("total=%d\n", cc);

	fprintf(output1, "rho\n");
	for (j = 0; j < 55; j++) {
		fprintf(output1, "%lf\n", model.rho[j]);
	}
	fprintf(output1, "label\n");
	for (j = 0; j < 11; j++) {
		fprintf(output1, "%d ", model.label[j]);
	}
	fprintf(output1, "\n");
	fprintf(output1, "nSV\n");
	for (j = 0; j < 11; j++) {
		fprintf(output1, "%d ", nSV[j]);
	}
	fprintf(output1, "\n");
	for (j = 0; j < NR_SV; j++) {
		fprintf(output1, "SV-No.%d\n", (j + 1));
		for (i = 0; i < 10; i++) {
			fprintf(output1, "%lf\n", model.sv_coef[i][ind[j]]);
		}
		for (i = 0; i < 10; i++) {
			fprintf(output1, "%lf\n", model.SV[ind[j]][i].value);
		}
	}	
	fclose(output1);
}

int main () {
	svm_load_model("model_revised.txt");
	print_model("model_reduced.txt");
	// printf(FORMAT, predict_sample(("testcase_200.txt")));
	// printf(FORMAT, round_real(2.38123453));
	// printf(FORMAT, min);
	// printf(FORMAT, max);
	return 0;
}
