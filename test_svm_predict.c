#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "svm.h"

int print_null(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;

struct svm_node *x;
int max_nr_attr = 64;

struct svm_model* model;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void predict(FILE *input, FILE *output)
{
	FILE *output1 = fopen ("tu/testcase_200.txt", "w");
	printf("in predict..\n");
  	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
	double *prob_estimates=NULL;
	int j;

	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);

			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			++i;
		}

		fprintf(output1, "%.0f\n", target_label); 
		int k = 0;		
		for (k = 0; k < i; k++) {
			fprintf(output1, "%lf\n", x[k].value);
		}
		
		x[i].index = -1;

		predict_label = svm_predict(model,x);
		fprintf(output,"%g\n",predict_label);
		
		if(predict_label == target_label) {
			//printf("correct=%d\n", correct);
      		++correct;
    	}
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
  	printf("i am here\n");
	if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
	{
		info("Mean squared error = %g (regression)\n",error/total);
		info("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	}
	else
		info("Accuracy = %g%% (%d/%d) (classification)\n",
			(double)correct/total*100,correct,total);
	fclose(output1);
}

int main() 
{
	FILE *input = fopen("tu/test-random-200.txt.scale", "r"); 
	FILE *output = fopen ("tu/test-random-200.txt.result", "w");
	FILE *output1 = fopen ("tu/model_reduced.txt", "w");
  	printf("before get model\n");
	model = svm_load_model("tu/train-random-1000.txt.scale.model");
  	printf("after get model\n");

	int i = 0, j = 0;
	fprintf(output1, "rho\n");
	for (j = 0; j < 10; j++) {
		fprintf(output1, "%lf\n", model->rho[j]);
	}
	fprintf(output1, "label\n");
	for (j = 0; j < 5; j++) {
		fprintf(output1, "%d ", model->label[j]);
	}
	fprintf(output1, "\n");
	fprintf(output1, "nSV\n");
	for (j = 0; j < 5; j++) {
		fprintf(output1, "%d ", model->nSV[j]);
	}
	fprintf(output1, "\n");
	for (j = 0; j < 539; j++) {
		fprintf(output1, "SV-No.%d\n", (j + 1));
		for (i = 0; i < 4; i++) {
			fprintf(output1, "%.15f\n", model->sv_coef[i][j]);
		}
		for (i = 0; i < 13; i++) {
			fprintf(output1, "%.7f\n", model->SV[j][i].value);
		}
	}	
	x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
	predict(input,output);
  	//svm_free_and_destroy_model(&model);
	//free(x);
	//free(line);
	fclose(input);
	fclose(output);
	fclose(output1);
	return 0;
}	