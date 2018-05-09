#include "ML.h"

#define start() t1 = chrono::high_resolution_clock::now();
#define end() t2 = chrono::high_resolution_clock::now();
#define print(msg) cout << msg << " time = " << (double)chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() / 1000 << " seconds" << endl;

namespace SecureML
{
	/****************************************************************************/
	Ciphertext ML::InnerProduct(Ciphertext* encZData, Ciphertext* encWData) {
		Ciphertext* encIPvec = new Ciphertext[params.cnum];
		for(long i = 0; i < params.cnum; i++) {
			scheme.modDownToAndEqual(encZData[i], encWData[i].logq);
			encIPvec[i] = scheme.mult(encZData[i], encWData[i]); //> 2 * wBits
			for(long j = 1; j < params.bBits; j++) {
				Ciphertext encrot = scheme.leftRotateByPo2(encIPvec[i], j);
				scheme.addAndEqual(encIPvec[i], encrot); //> 2 * wBits
			}
		}
		Ciphertext encIP = encIPvec[0];
		for(long i = 1; i < params.cnum; i++) {
			scheme.addAndEqual(encIP, encIPvec[i]); //> 2 * wBits
		}
		scheme.reScaleByAndEqual(encIP, params.wBits); //> wBits
		scheme.multByPolyAndEqual(encIP, auxpoly, params.pBits); //> wBits + pBits
		for (long i = 0; i < params.bBits; i++) {
			Ciphertext tmp = scheme.rightRotateByPo2(encIP, i);
			scheme.addAndEqual(encIP, tmp);
		}
		delete[] encIPvec;
		return encIP;
	}
	/****************************************************************************/
	void ML::Sigmoid(Ciphertext* encGrad, Ciphertext* encZData, Ciphertext& encIP, double gamma) {
		Ciphertext encIPsqr = scheme.square(encIP); //> 2 * wBits - 2 * aBits
		scheme.reScaleByAndEqual(encIPsqr, params.wBits); //> wBits - 2 * aBits
		scheme.addConstAndEqual(encIPsqr, degree3[1] / degree3[3], params.wBits - 2 * params.aBits);
		for (long i = 0; i < params.cnum; ++i)
		{
			encGrad[i] = scheme.multByConst(encZData[i], (gamma * degree3[3]), params.wBits + 3 * params.aBits);
			scheme.reScaleByAndEqual(encGrad[i], params.wBits); //> wBits + 3 * params.aBits
			if(encGrad[i].logq > encIP.logq) scheme.modDownToAndEqual(encGrad[i], encIP.logq);
			else scheme.modDownToAndEqual(encIP, encGrad[i].logq);
			scheme.multAndEqual(encGrad[i], encIP); //> 2 * wBits + 2 * params.aBits
			scheme.reScaleByAndEqual(encGrad[i], params.wBits); //> wBits + 2 * params.aBits
			scheme.multAndEqual(encGrad[i], encIPsqr); //> 2 * wBits
			scheme.reScaleByAndEqual(encGrad[i], params.wBits); //> wBits
			Ciphertext tmp = scheme.multByConst(encZData[i], (gamma * degree3[0]), params.wBits); //> 2 * wBits
			scheme.reScaleByAndEqual(tmp, params.wBits); //> wBits
			if(tmp.logq > encGrad[i].logq) scheme.modDownToAndEqual(tmp, encGrad[i].logq);
			else scheme.modDownToAndEqual(encGrad[i], tmp.logq);
			scheme.addAndEqual(encGrad[i], tmp);
		}
		for (long i = 0; i < params.cnum; ++i) {
			for (long l = params.bBits; l < params.sBits; ++l) {
				Ciphertext tmp = scheme.leftRotateByPo2(encGrad[i], l);
				scheme.addAndEqual(encGrad[i], tmp);
			}
		}
	}
	/****************************************************************************/
	void ML::EncryptzData(double** zData, long factorNum, long sampleNum) {
		long blockNum = params.sampleNum / params.blockSize;
		pool.exec_range(blockNum * params.cnum,
				[&](long first, long last) {
					for(long idx = first; idx < last; idx++) {
						long i = idx % blockNum;
						long j = idx / blockNum;
						std::complex<double>* pzData = new std::complex<double>[params.slots]();
						for(long k = 0; k < params.blockSize; k++) {
							for(long l = 0; l < params.batch; l++) {
								if((params.blockSize * i + k) >= sampleNum || (params.batch * j + l) >= factorNum) {
									pzData[params.batch * k + l] = 0;
								} else {
									pzData[params.batch * k + l].real(zData[params.blockSize * i + k][params.batch * j + l]);
								}
							}
						}
						Ciphertext encZData = scheme.encrypt(pzData, params.slots, params.wBits, params.logQ);
						SerializationUtils::writeCiphertext(encZData, "../encData/Ciphertext_" + std::to_string(i + 1) + "_" + std::to_string(j + 1) + ".txt");
						delete[] pzData;
					}
				}
			);
	}
	/****************************************************************************/
	void ML::Update(Ciphertext* encWData, Ciphertext* encVData, double gamma, double eta) {
		// Copy encWData and encVData to each thread //
		Ciphertext* encWDataCopyThread = new Ciphertext[pool.NumThreads() * params.cnum];
		Ciphertext* encVDataCopyThread = new Ciphertext[pool.NumThreads() * params.cnum];
		for(long i = 0; i < pool.NumThreads() * params.cnum; i++) {
			encWDataCopyThread[i] = encWData[i % params.cnum]; //> log scale = wBits
			encVDataCopyThread[i] = encVData[i % params.cnum]; //> log scale = wBits
		}

		long blockNum = params.sampleNum / params.blockSize;

		// Multi-Threaded Training //
		pool.exec_index(pool.NumThreads(), [&](long idx) {

			// 1. Pick Random blockID
			long blockID = rand() % blockNum;

			// 2. Read txt file and convert it to Ciphertext class
			Ciphertext* encZData = new Ciphertext[params.cnum]; //> wBits
			for(long i = 0; i < params.cnum; i++) {
				encZData[i] = SerializationUtils::readCiphertext("../encData/Ciphertext_" + std::to_string(blockID + 1) + "_" + std::to_string(i + 1) + ".txt");
				DecryptAndPrint("encZData_"+to_string(i+1), encZData[i]);
			}

			// 3. Update encWData and encVData (based on encZData)
			Ciphertext* encGrad = new Ciphertext[params.cnum];
			Ciphertext encIP = InnerProduct(encZData, encVDataCopyThread + idx * params.cnum);
			scheme.reScaleByAndEqual(encIP, params.pBits + params.aBits); //> wBits - aBits
			Sigmoid(encGrad, encZData, encIP, gamma);
			for(long i = 0; i < params.cnum; i++) {
				scheme.modDownToAndEqual(encVDataCopyThread[idx * params.cnum + i], encGrad[i].logq);
				Ciphertext ctmpw = scheme.sub(encVDataCopyThread[idx * params.cnum + i], encGrad[i]); //> wBits
				encVDataCopyThread[idx * params.cnum + i] = scheme.multByConst(ctmpw, 1. - eta, params.pBits); //> wBits + pBits
				scheme.reScaleByAndEqual(encVDataCopyThread[idx * params.cnum + i], params.pBits); //> wBits
				scheme.multByConstAndEqual(encWDataCopyThread[idx * params.cnum + i], eta, params.pBits); //> wBits + pBits
				scheme.reScaleByAndEqual(encWDataCopyThread[idx * params.cnum + i], params.pBits); //> wBits
				scheme.modDownToAndEqual(encWDataCopyThread[idx * params.cnum + i], encVDataCopyThread[idx * params.cnum + i].logq);
				scheme.addAndEqual(encVDataCopyThread[idx * params.cnum + i], encWDataCopyThread[idx * params.cnum + i]);
				encWDataCopyThread[idx * params.cnum + i] = ctmpw;
				scheme.modDownToAndEqual(encWDataCopyThread[idx * params.cnum + i], encVDataCopyThread[idx * params.cnum + i].logq);
			}

			delete[] encZData;
			delete[] encGrad;
		});

		// Compute average of each results from each thread and update //
		for(long i = 0; i < params.cnum; i++) {
			encWData[i] = encWDataCopyThread[i];
			encVData[i] = encVDataCopyThread[i];
			for(long j = 1; j < pool.NumThreads(); j++) {
				scheme.addAndEqual(encWData[i], encWDataCopyThread[j * params.cnum + i]);
				scheme.addAndEqual(encVData[i], encVDataCopyThread[j * params.cnum + i]);
			}
			scheme.divByPo2AndEqual(encWData[i], (long)log2(pool.NumThreads()));
			scheme.divByPo2AndEqual(encVData[i], (long)log2(pool.NumThreads()));
		}

		delete[] encWDataCopyThread;
		delete[] encVDataCopyThread;

	}
	/****************************************************************************/
	void ML::Training(Ciphertext* encWData, long factorNum) {
		chrono::high_resolution_clock::time_point t1, t2;
		
		string FILE("../data/MIMIC.csv");
		double gamma, eta;
		double alpha0, alpha1;

		Ciphertext* encVData = new Ciphertext[params.cnum];
		for(long i = 0; i < params.cnum; i++) {
			encWData[i] = scheme.encryptZeros(params.batch, params.wBits, params.logQ); //> wBits
			encVData[i] = scheme.encryptZeros(params.batch, params.wBits, params.logQ); //> wBits
		}

		alpha0 = 0.01;
		alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0;
		gamma = params.alpha / params.blockSize;

		for(long iter = 0; iter < params.iterNum; iter++) {
			///////////////////////////////////////////////////////
			cout << iter + 1 << "-th iteration started !!!" << endl;
			///////////////////////////////////////////////////////

			eta = (1 - alpha0) / alpha1;

			start();
			Update(encWData, encVData, gamma, eta);
			end(); print("update");

			double* dwData = new double[factorNum];
			DecryptwData(dwData, encWData, factorNum);
			SecureML::testAUROC(FILE, dwData);

			alpha0 = alpha1;
			alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0;

			if(iter % params.iterPerBoot == params.iterPerBoot - 1 && iter != params.iterNum - 1)
			{
				cout << "Bootstrapping START!!!" << endl;
				start();
				pool.exec_index(params.cnum * 2, [&](long idx) {
					long index;
					if(idx % 2 == 0) { // W
						index = idx / 2;
						encWData[index].slots = params.batch;
						scheme.bootstrapAndEqual(encWData[index], params.logq, params.logQBoot, params.logT, params.logI);
						encWData[index].slots = params.slots;
					} else { // V
						index = (idx - 1) / 2;
						encVData[index].slots = params.batch;
						scheme.bootstrapAndEqual(encVData[index], params.logq, params.logQBoot, params.logT, params.logI);
						encVData[index].slots = params.slots;
					}
				});
				end(); print("bootstrapping");
				cout << "Bootstrapping END!!!" << endl;
				DecryptwData(dwData, encWData, factorNum);
				SecureML::testAUROC(FILE, dwData);
			}
			delete[] dwData;
		}
		delete[] encVData;
	}
	/****************************************************************************/
	void ML::DecryptwData(double* wData, Ciphertext* encWData, long factorNum) {
		for (long i = 0; i < (params.cnum - 1); ++i) {
			complex<double>* dcw = scheme.decrypt(sk, encWData[i]);
			for (long j = 0; j < params.batch; ++j) {
				wData[params.batch * i + j] = dcw[j].real();
			}
			delete[] dcw;
		}
		complex<double>* dcw = scheme.decrypt(sk, encWData[params.cnum-1]);
		long rest = factorNum - params.batch * (params.cnum - 1);
		for (long j = 0; j < rest; ++j) {
			wData[params.batch * (params.cnum - 1) + j] = dcw[j].real();
		}
		delete[] dcw;
	}
	/****************************************************************************/
	void ML::DecryptAndPrint(string msg, Ciphertext cipher) {
		complex<double>* dp = scheme.decrypt(sk, cipher);
		cout << msg + " = ";
		for(long i = 0; i < 10; i++) {
			cout << dp[i].real() << ",";
		}
		cout << endl;
	}
}

