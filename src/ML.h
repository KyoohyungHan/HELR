#ifndef SRC_ML_H_
#define SRC_ML_H_

#include <NTL/BasicThreadPool.h>

#include <complex.h>
#include <vector>
#include <string>

#include "Scheme.h"
#include "Params.h"
#include "Ciphertext.h"
#include "SerializationUtils.h"

#include "functions.h"

using namespace std;

static double degree3[4] = {-0.5,0.15012,0.0,-0.001593}; //-> -0.5 + 0.15012 * X - 0.001593 * X^3

namespace SecureML {

	class ML {

	private:

		SecretKey& sk;

		BasicThreadPool& pool;

		Ciphertext InnerProduct(Ciphertext* encZData, Ciphertext* encWData);

		void Sigmoid(Ciphertext* encGrad, Ciphertext* encZData, Ciphertext& encIP, double gamma);

	public:

		// Address of scheme //
		Scheme& scheme;

		// SecureML parameter //
		Params& params;

		ZZX auxpoly;

		ML(Scheme& scheme, Params& params, BasicThreadPool& pool, SecretKey& sk) : scheme(scheme), pool(pool), params(params), sk(sk) {

			// Generate auxpolynomial which is encoding of some special vector //
			complex<double>* pvals = new complex<double>[params.slots]();
			for (long i = 0; i < params.slots; i += params.batch) {
				pvals[i].real(1.0);
			}
			auxpoly = scheme.context.encode(pvals, params.slots, params.pBits);
			delete[] pvals;

		}

		// Encrypt zData and save encrypted value in Hard Drive //
		void EncryptzData(double** zData, long factorNum, long sampleNum);

		// Run secure training algorithm using encrypted value (saved in Hard Drive) //
		void Update(Ciphertext* encWData, Ciphertext* encVData, double gamma, double eta);

		// Training by repeat update process params.numIter times //
		void Training(Ciphertext* encWData, long factorNum);

		// Decrypt encrypted wData //
		void DecryptwData(double* wData, Ciphertext* encWData, long factorNum);

		// Decrypt and print //
		void DecryptAndPrint(string msg, Ciphertext cipher);

	};
}


#endif /* SRC_ML_H_ */
