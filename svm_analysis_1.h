#ifndef _LIBSVM_H
#define _LIBSVM_H

#define LIBSVM_VERSION 316

#ifdef __cplusplus
extern "C" {
#endif

extern int libsvm_version;

struct svm_node 12b
{
	int index;
	double value;
};

struct svm_problem
{
	int l;
	double *y;
	struct svm_node **x;
};

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

struct svm_parameter
{
	int svm_type;    4B
	int kernel_type; 4B
	int degree;	/* for poly */ 4B
	double gamma;	/* for poly/rbf/sigmoid */ 8B
	double coef0;	/* for poly/sigmoid */     8B

	/* these are for training only */
	double cache_size; /* in MB */				8B
	double eps;	/* stopping criteria */			8B
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */ 
	int nr_weight;		/* for C_SVC */         4B
	int *weight_label;	/* for C_SVC */			0
	double* weight;		/* for C_SVC */         0
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */  4B
	double p;	/* for EPSILON_SVR */                    4B        
	int shrinking;	/* use the shrinking heuristics */   4B 
	int probability; /* do probability estimates */      4B
}; total = 64B

//
// svm_model
// 
struct svm_model
{
	struct svm_parameter param;	64B 
	int nr_class;
	int l;
	struct svm_node **SV;	
	double **sv_coef;
	double *rho;
	/* for classification only */
    
	int *label;
	int *nSV;
}; 

struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

int svm_save_model(const char *model_file_name, const struct svm_model *model);
struct svm_model *svm_load_model(const char *model_file_name);

int svm_get_svm_type(const struct svm_model *model);
int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);
void svm_get_sv_indices(const struct svm_model *model, int *sv_indices);
int svm_get_nr_sv(const struct svm_model *model);
double svm_get_svr_probability(const struct svm_model *model);

double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
double svm_predict(const struct svm_model *model, const struct svm_node *x);
double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

void svm_free_model_content(struct svm_model *model_ptr);
void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
void svm_destroy_param(struct svm_parameter *param);

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
int svm_check_probability_model(const struct svm_model *model);

void svm_set_print_string_function(void (*print_func)(const char *));

#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */
