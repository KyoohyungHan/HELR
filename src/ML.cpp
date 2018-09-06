#include "ML.h"

#define start() t1 = chrono::high_resolution_clock::now();
#define end() t2 = chrono::high_resolution_clock::now();
#define print(msg) cout << msg << " time = " << (double)chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() / 1000 << " seconds" << endl;

namespace SecureML
{
	/****************************************************************************/
	// Inner Product consumes (wBits + pBits) - bit of ciphertext modulus //
	Ciphertext ML::InnerProduct(Ciphertext* encZData, Ciphertext* encVData) {
		////////////////////////////////
		Ciphertext* encIPvec = new Ciphertext[params.cnum];
		pool.exec_index(params.cnum, [&](long i) {
			Ciphertext tmp = scheme.modDownTo(encZData[i], encVData[i].logq);
			encIPvec[i] = scheme.mult(tmp, encVData[i]); //> logp: 2 * wBits
			for(long j = 0; j < params.bBits; j++) {
				Ciphertext encrot = scheme.leftRotateFast(encIPvec[i], 1 << j);
				scheme.addAndEqual(encIPvec[i], encrot); //> logp: 2 * wBits
			}
		});
		////////////////////////////////
		Ciphertext encIP = encIPvec[0];
		for(long i = 1; i < params.cnum; i++) {
			scheme.addAndEqual(encIP, encIPvec[i]); //> logp: 2 * wBits
		}
		////////////////////////////////> Multiply Dummy Matrix
		scheme.multByPolyAndEqual(encIP, dummy, params.pBits); //> logp: 2 * wBits + pBits
		////////////////////////////////
		for(long i = 0; i < params.bBits; i++) {
			Ciphertext tmp = scheme.rightRotateFast(encIP, 1 << i);
			scheme.addAndEqual(encIP, tmp);
		}
		//> Re-scaling the scaling factor
		scheme.reScaleByAndEqual(encIP, params.pBits + params.wBits); //> logp: wBits, bitDown: wBits + pBits
		delete[] encIPvec;
		return encIP;
	}
	/****************************************************************************/
	// Sigmoid consumes (2 * wBits)-bit of ciphertext modulus //
	void ML::Sigmoid(Ciphertext* encGrad, Ciphertext* encZData, Ciphertext& encIP, double gamma) {
		Ciphertext encIPsqr = scheme.square(encIP); //> logp: 2 * wBits
		scheme.addConstAndEqual(encIPsqr, degree3[1] / degree3[3], 2 * params.wBits);
		scheme.reScaleByAndEqual(encIPsqr, params.wBits); //> logp: wBits, bitDown: wBits
		////////////////////////////////
		pool.exec_index(params.cnum, [&](long i) {
			encGrad[i] = scheme.multByConst(encZData[i], (gamma * degree3[3]), 2 * params.wBits); //> logq: logQ + wBits
			scheme.reScaleByAndEqual(encGrad[i], 2 * params.wBits); //> logp: wBits, logq: logQ - pBits
			scheme.modDownToAndEqual(encGrad[i], encIP.logq);
			scheme.multAndEqual(encGrad[i], encIP); //> logp: 2 * wBits
			scheme.reScaleByAndEqual(encGrad[i], params.wBits); //> logp: wBits, bitDown: wBits
			scheme.multAndEqual(encGrad[i], encIPsqr); //> logp: 2 * wBits
			scheme.reScaleByAndEqual(encGrad[i], params.wBits); //> logp: wBits, bitDown: 2 * wBits
			Ciphertext tmp = scheme.multByConst(encZData[i], (gamma * degree3[0]), 2 * params.wBits); //> 2 * wBits
			scheme.reScaleByAndEqual(tmp, 2 * params.wBits); //> logp: wBits, logq: logQ - wBits
			scheme.modDownToAndEqual(tmp, encGrad[i].logq);
			scheme.addAndEqual(encGrad[i], tmp); //> bitDown: 2 * wBits
			for (long l = params.bBits; l < params.sBits; ++l) {
				Ciphertext tmp = scheme.leftRotateFast(encGrad[i], 1 << l);
				scheme.addAndEqual(encGrad[i], tmp);
			}
		});
	}
	/****************************************************************************/
	void ML::plainInnerProduct(double* ip, double** zData, double* vData, long factorNum, long sampleNum) {
		for(long i = 0; i < sampleNum; i++) {
			ip[i] = innerproduct(vData, zData[i], factorNum);
		}
	}
	/****************************************************************************/
	void ML::plainSigmoid(double* grad, double** zData, double* ip, double gamma, long factorNum, long sampleNum) {
		for(long i = 0; i < sampleNum; i++) {
			//double tmp = (degree3[0] + degree3[1] * ip[i] + degree3[3] * pow(ip[i], 3)); //> ~ 1 / (1 + exp(x))
			double tmp = (1. / (1. + exp(ip[i])));
			tmp *= gamma;
			for(long j = 0; j < factorNum; j++) {
				grad[j] += tmp * zData[i][j];
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
						Ciphertext encZData = scheme.encrypt(pzData, params.slots, params.wBits, params.logQ + params.wBits);
						SerializationUtils::writeCiphertext(encZData, "../encData/Ciphertext_" + std::to_string(i + 1) + "_" + std::to_string(j + 1) + ".txt");
						delete[] pzData;
					}
				}
			);
	}
	/****************************************************************************/
	void ML::Update(Ciphertext* encWData, Ciphertext* encVData, double gamma, double eta, long blockID) {
		// Read txt file and convert it to Ciphertext class //
		Ciphertext* encZData = new Ciphertext[params.cnum]; //> wBits
		for(long i = 0; i < params.cnum; i++) {
			encZData[i] = SerializationUtils::readCiphertext("../encData/Ciphertext_" + std::to_string(blockID + 1) + "_" + std::to_string(i + 1) + ".txt");
		}
		// Update encWData and encVData (based on encZData) //
		Ciphertext* encGrad = new Ciphertext[params.cnum];
		Ciphertext encIP = InnerProduct(encZData, encVData); //> bitDown: wBits + pBits
		Sigmoid(encGrad, encZData, encIP, gamma); //> bitDown: 3 * wBits + 2 * pBits
		// Update encWData and encVData //
		pool.exec_index(params.cnum, [&](long i) {
			Ciphertext tmp1 = scheme.modDownTo(encVData[i], encGrad[i].logq);
			scheme.addAndEqual(tmp1, encGrad[i]);
			Ciphertext tmp2 = scheme.multByConst(encWData[i], eta, params.wBits + params.pBits);
			scheme.reScaleByAndEqual(tmp2, params.wBits + params.pBits);
			encVData[i] = scheme.multByConst(tmp1, (1. - eta), params.pBits);
			scheme.reScaleByAndEqual(encVData[i], params.pBits);
			scheme.modDownToAndEqual(tmp2, encVData[i].logq);
			scheme.addAndEqual(encVData[i], tmp2);
			encWData[i] = tmp1;
			scheme.modDownToAndEqual(encWData[i], encVData[i].logq);
		});
		delete[] encZData;
		delete[] encGrad;
	}
	/****************************************************************************/
	void ML::Training(Ciphertext* encWData, long factorNum, long sampleNum, double* wData, double** zData) {
		chrono::high_resolution_clock::time_point t1, t2;
		
		double gamma, eta;
		double alpha0, alpha1;

		Ciphertext* encVData = new Ciphertext[params.cnum];
		double* zeroVec = new double[params.slots]();
		for(long i = 0; i < params.cnum; i++) {
			encWData[i] = scheme.encrypt(zeroVec, params.slots, params.wBits, params.logQ); //> wBits
			encVData[i] = scheme.encrypt(zeroVec, params.slots, params.wBits, params.logQ); //> wBits
		}

		double* vData = new double[params.factorNum]();
		for(long i = 0; i < params.factorNum; i++) {
			wData[i] = 0.;
		}

		alpha0 = 0.01;
		alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0;
		
		gamma = params.alpha / params.blockSize;
		long blockNum = params.sampleNum / params.blockSize; 

		double* dwData = new double[factorNum];

		ofstream auroc_enc("auroc_enc_" + to_string(params.numThread) + ".csv");
		ofstream auroc_ptxt("auroc_ptxt_" + to_string(params.numThread) + ".csv");
		ofstream accuracy_enc("accuracy_enc_" + to_string(params.numThread) + ".csv");
		ofstream accuracy_ptxt("accuracy_ptxt_" + to_string(params.numThread) + ".csv");

		double auc, accuracy;

		for(long iter = 0; iter < params.iterNum; iter++) {
			///////////////////////////////////////////////////////
			cout << endl;
			cout << iter + 1 << "-th iteration started !!!" << endl;
			///////////////////////////////////////////////////////

			eta = (1 - alpha0) / alpha1;

			long blockID = rand() % blockNum;

			cout << "** un-encrypted" << endl;
			plainUpdate(wData, vData, zData, gamma, eta, factorNum, sampleNum, blockID);
			SecureML::testAUROC(auc, accuracy, params.path_to_test_file, dwData, params.isfirst);
			accuracy_ptxt << iter + 1 << ", " << accuracy << endl;
			auroc_ptxt << iter + 1 << ", " << auc << endl;

			cout << "** encrypted" << endl;
			start();
			Update(encWData, encVData, gamma, eta, blockID);
			end(); print("update");

			DecryptwData(dwData, encWData, factorNum);
			SecureML::testAUROC(auc, accuracy, params.path_to_test_file, dwData, params.isfirst);
			accuracy_enc << iter + 1 << ", " << accuracy << endl;
			auroc_enc << iter + 1 << ", " << auc << endl;			
			
			alpha0 = alpha1;
			alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0;

			if(iter % params.iterPerBoot == params.iterPerBoot - 1 && iter < params.iterNum - 1) {
				
				cout << "\nBootstrapping START!!!" << endl;
				start();
				pool.exec_index(params.cnum, [&](long i) {
					encWData[i].n = params.batch;
					scheme.bootstrapAndEqual(encWData[i], params.logq, params.logQBoot, params.logT, params.logI);
					encWData[i].n = params.slots;
				});
				pool.exec_index(params.cnum, [&](long i) {
					encVData[i].n = params.batch;
					scheme.bootstrapAndEqual(encVData[i], params.logq, params.logQBoot, params.logT, params.logI);
					encVData[i].n = params.slots;
				
				});
				end(); print("bootstrapping");
				cout << "Bootstrapping END!!!" << endl;
				DecryptwData(dwData, encWData, factorNum);
				SecureML::testAUROC(auc, accuracy, params.path_to_test_file, dwData, params.isfirst);
			}
		}
		auroc_enc.close();
		auroc_ptxt.close();
		accuracy_enc.close();
		accuracy_ptxt.close();
		delete[] dwData;
		delete[] encVData;
	}
	/****************************************************************************/
	void ML::plainUpdate(double* wData, double* vData, double** zData, double gamma, double eta, long factorNum, long sampleNum, long blockID) {
		// Select zData Block //
		double** zBlockData = new double*[params.blockSize];
		for(long i = 0; i < params.blockSize; i++) {
			zBlockData[i] = new double[params.factorNum];
			for(long j = 0; j < params.factorNum; j++) {
				if((blockID * params.blockSize + i) < sampleNum && j < factorNum) zBlockData[i][j] = zData[blockID * params.blockSize + i][j];
				else zBlockData[i][j] = 0.;
			}
		}
		// Allocate two array //
		double* grad = new double[params.factorNum]();
		double* ip = new double[params.blockSize]();
		// Compute Inner Product and Sigmoid //
		plainInnerProduct(ip, zBlockData, vData, params.factorNum, params.blockSize);
		plainSigmoid(grad, zBlockData, ip, gamma, params.factorNum, params.blockSize);
		// Update wData using vData //
		for(long i = 0; i < params.factorNum; i++) {
			double tmp1 = vData[i] + grad[i];
			double tmp2 = eta * wData[i];
			vData[i] = tmp1 * (1. - eta) + tmp2;
			wData[i] = tmp1;
		}
		delete[] grad;
		delete[] ip;
	}
	/****************************************************************************/
	void ML::plainTraining(double* wData, double** zData, long factorNum, long sampleNum) {
		chrono::high_resolution_clock::time_point t1, t2;
		
		double gamma, eta;
		double alpha0, alpha1;

		double* vData = new double[params.factorNum]();
		for(long i = 0; i < params.factorNum; i++) {
			wData[i] = 0.;
		}

		alpha0 = 0.01;
		alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0;

		gamma = params.alpha / params.blockSize;
		long blockNum = params.sampleNum / params.blockSize; 

		//ofstream acc("accuracy" + to_string(params.numThread) + "_ptxt_deg3.csv");
		ofstream acc("accuracy" + to_string(params.numThread) + "_ptxt_sigmoid.csv");

		double auc, accuracy;

		for(long iter = 0; iter < params.iterNum; iter++) {
			///////////////////////////////////////////////////////
			cout << endl;
			cout << iter + 1 << "-th iteration started (plain)!!!" << endl;
			///////////////////////////////////////////////////////

			eta = (1 - alpha0) / alpha1;

			srand(iter);

			start();
			plainUpdate(wData, vData, zData, gamma, eta, factorNum, sampleNum, rand() % blockNum);
			end(); print("plain update");

			testAUROC(auc, accuracy, params.path_to_test_file, wData, params.isfirst);

			acc << iter + 1 << ", " << accuracy << endl;

			alpha0 = alpha1;
			alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0;
		}
		acc.close();
	}
	/****************************************************************************/
	void ML::DecryptwData(double* wData, Ciphertext* encWData, long factorNum) {
		for (long i = 0; i < params.cnum; ++i) {
			complex<double>* dcw = scheme.decrypt(sk, encWData[i]);
			for (long j = 0; j < params.batch; ++j) {
				if(params.batch * i + j < factorNum)
					wData[params.batch * i + j] = dcw[j].real();
			}
			delete[] dcw;
		}
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

