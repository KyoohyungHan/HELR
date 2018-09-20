#ifndef SRC_PARAMS_H_
#define SRC_PARAMS_H_

#include <cmath>
#include <string>
#include <algorithm>

using namespace std;

namespace SecureML {

	class Params {

	public:

		long logQ, logQBoot, logq;

		long logN;

		long logI, logT;

		long wBits, pBits, lBits;

		long kdeg;

		long iterNum, iterPerBoot;

		long factorNum, sampleNum;

		long blockSize;

		long fNumBits, sNumBits;

		long slots, cnum;

		long batch, bBits, sBits;

		long numThread;

		bool isfirst;

		string path_to_file;

		string path_to_test_file;

		double alpha;

		Params();

		Params(long factorNum, long sampleNum, long iterNum, double alpha, long numThread)
		{
			this->factorNum = 1 << (long)ceil(log2(factorNum));
			this->iterNum = iterNum;
			this->numThread = numThread;
			this->alpha = alpha;

			// Set precision bits //
			wBits = 40; pBits = 15; lBits = 5;

			// Set Iteration Number per bootstrapping //
			iterPerBoot = 3;

			// Set degree of approximate polynomial //
			kdeg = 3;

			// Compute logQ //
			logQ = wBits + lBits + iterPerBoot * (3 * wBits + 2 * pBits);

			// Compute logQBoot for bootstrapping //
			logq = wBits + lBits;
			logI = 4; logT = 3;
			long bitForBoot = 16 + logT + logI + (logI + logT + 6) * (logq + logI);
			logQBoot = logQ + bitForBoot;

			// Compute proper logN for security parameter 80-bit //
			long NBnd = ceil(logQBoot * (80 + 110) / 3.6);
			double logNBnd = log2((double)NBnd);
			logN = (long)ceil(logNBnd);
			if(logN > 16) cerr << "We recommand you to use smaller iterPerBoot!!!" << endl;

			// Compute Best blockSize for given numThread //
			fNumBits = (long)ceil(log2(factorNum));
			sNumBits = (long)ceil(log2(sampleNum));

			// Select other parameters //
			cnum = numThread;
			batch = this->factorNum / cnum;
			slots = 1 << (logN - 1);
			blockSize = slots / batch;
			this->sampleNum = (sampleNum / blockSize) * blockSize;
			if(sampleNum % blockSize != 0) this->sampleNum += blockSize;
			bBits = (long)ceil(log2(batch));
			sBits = (long)ceil(log2(slots));
		
			// Print parameters //
			cout << "***********************************" << endl;
			cout << "Secure Machine Learning Parameters" << endl;
			cout << "***********************************" << endl;
			cout << "- factorNum = " << this->factorNum << ", sampleNum = " << this->sampleNum << ", approximate degree = " << kdeg << endl;
			cout << "- Iteration Number = " << this->iterNum << ", mini-batch Block Size = " << blockSize << ", Learning Rate = " << this->alpha << endl;
			cout << "- logN = " << this->logN << ", logQ = " << logQ << ", logQBoot = " << logQBoot << ", batch = " << batch << ", cnum = " << cnum << endl;
			cout << "- logq = " << this->logq << ", logI = " << this->logI << ", logT = " << this->logT << endl;

		}
	};
}

#endif /* SRC_PARAMS_H_ */
