#ifndef SRC_FUNCTIONS_H_
#define SRC_FUNCTIONS_H_

#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

using namespace std;

namespace SecureML {

	double** zDataFromFile(string path, long& factorDim, long& sampleDim, bool isfirst = true);
	
	void shuffleZData(double** zData, long factorDim, long sampleDim);

	void normalizeZData(double** zData, long factorDim, long sampleDim);	
	
	double innerproduct(double* vec1, double* vec2, long size);

	void testAUROC(double& auc, double& accuracy, double** zData, long factorDim, long sampleDim, double* wData, bool isfirst = true);

}

#endif /* SRC_FUNCTIONS_H_ */
