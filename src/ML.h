#ifndef SRC_ML_H_
#define SRC_ML_H_

#include <NTL/BasicThreadPool.h>
#include <thread>

#include <complex.h>
#include <vector>
#include <string>

#include "HEAAN.h"
#include "Params.h"
#include "functions.h"

using namespace std;

//static double degree3[4] = {0.5,-0.15012,0.0,0.001593}; //> ~ 1 / (1 + exp(x)) (LSFitting with bnd [-8,8])
static double degree3[4] = {0.5,-0.0843,0.0,0.0002}; //> ~ 1/ (1 + exp(x)) (LSFitting with bnd [-16,16])

namespace SecureML {

	class ML {

	private:

		SecretKey& sk;

		BasicThreadPool& pool;

		Ciphertext InnerProduct(Ciphertext* encZData, Ciphertext* encVData);

		void Sigmoid(Ciphertext* encGrad, Ciphertext* encZData, Ciphertext& encIP, double gamma);

		void plainInnerProduct(double* ip, double** zData, double* vData, long factorNum, long sampleNum);

		void plainSigmoid(double* encGrad, double** zData, double* ip, double gamma, long factorNum, long sampleNum);

	public:

		// Address of scheme //
		Scheme& scheme;

		// SecureML parameter //
		Params& params;

		ZZ* dummy;

		ML(Scheme& scheme, Params& params, BasicThreadPool& pool, SecretKey& sk) : scheme(scheme), pool(pool), params(params), sk(sk) {

			// Generate dummy polynomial which is encoding of some special vector //
			complex<double>* pvals = new complex<double>[params.slots]();
			for (long i = 0; i < params.slots; i += params.batch) {
				pvals[i].real(1.0);
			}
			dummy = new ZZ[scheme.ring.N];
			scheme.ring.encode(dummy, pvals, params.slots, params.pBits);
			delete[] pvals;

		}

		// Encrypt zData and save encrypted value in Hard Drive //
		void EncryptzData(double** zData, long factorNum, long sampleNum);

		// Run secure training algorithm using encrypted value (saved in Hard Drive) //
		void Update(Ciphertext* encWData, Ciphertext* encVData, double gamma, double eta, long blockID);

		// Training by repeating update process params.numIter times //
		void Training(Ciphertext* encWData, long factorNum, long sampleNum, double* wData, double** zData);

		// Run training algorithm using un-enrypted data //
		void plainUpdate(double* wData, double* vData, double** zData, double gamma, double eta, long factorNum, long sampleNum, long blockID);

		// Training by repeating plainUpdate process parms.numIter times //
		void plainTraining(double* wData, double** zData, long factorNum, long sampleNum);

		// Decrypt encrypted wData //
		void DecryptwData(double* wData, Ciphertext* encWData, long factorNum);

		// Decrypt and save //
		void DecryptwDataAndSave(string fileName, Ciphertext* encWData, long factorNum);

		// Decrypt and print //
		void DecryptAndPrint(string msg, Ciphertext cipher);

	};
}


#endif /* SRC_ML_H_ */
