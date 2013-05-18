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
	printf("in predict..\n");
  	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int svm_type=NU_SVC;
	int nr_class=11;
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
		x[i].index = -1;

		predict_label = svm_predict(model,x);
		fprintf(output,"%g\n",predict_label);

		if(predict_label == target_label) {
			printf("correct=%d\n", correct);
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
}

int main() 
{
	FILE *input = fopen("vowel.scale.t.short", "r"); 
	FILE *output = fopen ("vowel.scale.t.result.short", "w");
  	printf("before get model\n");
	model = svm_load_model("vowel.scale.model");
  	printf("after get model\n");
	x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
	predict(input,output);

  	//svm_free_and_destroy_model(&model);
	//free(x);
	//free(line);
	fclose(input);
	fclose(output);
	return 0;
}	