#ifndef SRC_PARAMS_H_
#define SRC_PARAMS_H_

#include <cmath>
#include <algorithm>

using namespace std;

namespace SecureML {

	class Params {

	public:

		long logQ, logQBoot, logq;

		long logN;

		long logI, logT;

		long wBits, pBits, lBits, aBits;

		long kdeg, kBits;

		long iterNum, iterPerBoot;

		long factorNum, sampleNum;

		long blockSize;

		long fNumBits, sNumBits;

		long slots, cnum;

		long batch, bBits, sBits;

		long numThread;

		double alpha;

		Params();

		Params(long factorNum, long sampleNum, long iterNum, double alpha, long numThread)
		{
			this->factorNum = 1 << (long)ceil(log2(factorNum));
			this->sampleNum = 1 << (long)ceil(log2(sampleNum));
			this->iterNum = iterNum;
			this->numThread = numThread;
			this->alpha = alpha;

			// Set precision bits //
			wBits = 30; pBits = 20;
			lBits = 5; aBits = 5;

			// Set Iteration Number per bootstrapping //
			iterPerBoot = 4;

			// Set degree of approximate polynomial //
			kdeg = 3; kBits = (long)ceil(log2(kdeg));

			// Compute logQ //
			logQ = (wBits + lBits) + iterPerBoot * ((kBits + 1) * wBits + 2 * pBits + aBits + (long)ceil(log2(numThread)));

			// Compute logQBoot for bootstrapping //
			logq = wBits + 5;
			logI = 4; logT = 4;
			long bitForBoot = logN + (logI + logT + 5) * logq + (logI + logT + 6) * logI + logT + logq + 20;
			logQBoot = logQ + bitForBoot;

			// Compute proper logN for security parameter 80-bit //
			long NBnd = ceil(logQBoot * (80 + 110) / 3.6);
			double logNBnd = log2((double)NBnd);
			logN = (long)ceil(logNBnd);

			// Compute Best blockSize for given numThread //
			fNumBits = (long)ceil(log2(factorNum));
			sNumBits = 0;
			while(1)
			{
				if((numThread == 1 && cnum == numThread + 1) || sNumBits > (long)ceil(log2(sampleNum))){
					sNumBits--;
					bBits = min(logN - 1 - sNumBits, fNumBits);
					batch = 1 << bBits;
					sBits = sNumBits + bBits;
					slots =  1 << sBits;
					cnum = (long)ceil((double)(1 << fNumBits) / batch);
					break;
				}
				sNumBits++;
				bBits = min(logN - 1 - sNumBits, fNumBits);
				batch = 1 << bBits;
				sBits = sNumBits + bBits;
				slots =  1 << sBits;
				cnum = (long)ceil((double)(1 << fNumBits) / batch);
				if(2 * cnum == numThread) break;
			}
			blockSize = 1 << sNumBits;

			// Increase Iteration Number by mini-batch //
			this->iterNum *= (long)ceil((double)(sampleNum) / (numThread * blockSize));

			// Print parameters //
			cout << "***********************************" << endl;
			cout << "Secure Machine Learning Parameters" << endl;
			cout << "***********************************" << endl;
			cout << "- factorNum = " << this->factorNum << ", sampleNum = " << this->sampleNum << ", approximate degree = " << kdeg << endl;
			cout << "- Iteration Number = " << this->iterNum << ", mini-batch Block Size = " << blockSize << ", Learning Rate = " << this->alpha << endl;
			cout << "- logN = " << this->logN << ", logQ = " << logQ << ", logQBoot = " << logQBoot << ", batch = " << batch << ", cnum = " << cnum << endl;

		}
	};
}

#endif /* SRC_PARAMS_H_ */
