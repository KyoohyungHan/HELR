#include <iostream>
#include <stdlib.h>
#include <chrono>

#include <NTL/BasicThreadPool.h>

#include "Context.h"
#include "SecretKey.h"
#include "Scheme.h"

#include "Params.h"
#include "ML.h"
#include "functions.h"

chrono::high_resolution_clock::time_point t1, t2;

#define START() t1 = chrono::high_resolution_clock::now();
#define END() t2 = chrono::high_resolution_clock::now();
#define PRINTTIME(msg) cout << msg << " time = " << (double)chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() / 1000 << " seconds" << endl;

using namespace std;
using namespace NTL;

int main() {

	/* Construct BasicThreadPool and set the number of thread
	 * */
	long numThread = 8;
	BasicThreadPool pool(numThread);

	/* Read Data file and shuffle and normalize it
	 * */
	long factorNum, sampleNum;
	string FILE("data/MIMIC.csv");
	double** zData = SecureML::zDataFromFile(FILE, factorNum, sampleNum);
	SecureML::shuffleZData(zData, factorNum, sampleNum);
	SecureML::normalizeZData(zData, factorNum, sampleNum);

	/* Params(long factorNum, long sampleNum, long iterNum, double alpha, long numThread)
	 * alpha : learning rate (0.001 to 1.0 values are used in general
	 * iterNum does not need to be large because we use mini-batch technique + optimized gradient decent
	 */
	SecureML::Params params(factorNum, sampleNum, 20, 1.0, pool.NumThreads());

	/* Key Generation for HEAANBOOT library
	 * If params.iterNum is larger than params.iterNumPerBoot, this will generate public key for bootstrapping
	 * */
	START();
	Context context(params.logN, params.logQBoot);
	SecretKey sk(params.logN);
	Scheme scheme(sk, context);
	scheme.addLeftRotKeys(sk);
	scheme.addRightRotKeys(sk);
	scheme.addBootKey(sk, params.bBits, params.logq + params.logI);
	END(); PRINTTIME("\n - KeyGen");

	/* Simple Constructor for secureML class
	 * This will just pass some address of scheme, params, and pool
	 * */
	SecureML::ML secureML(scheme, params, pool, sk);

	/* Encrypt zData using public key (this part can be faster in symmetric key version)
	 * To prepare for large number of data case, encryption of zData will be saved in encData/ folder with .txt files
	 * */
	START();
	secureML.EncryptzData(zData, factorNum, sampleNum);
	END(); PRINTTIME(" - Encrypt");

	/* Training using zData which are saved as txt file
	 * */
	START();
	Ciphertext* encWData = new Ciphertext[params.cnum];
	secureML.Training(encWData, factorNum);
	END(); PRINTTIME("\n - Training");

	/* Decrypt encWData using secretkey sk
	 * */
	START();
	double* dwData = new double[factorNum]();
	secureML.DecryptwData(dwData, encWData, factorNum);
	END(); PRINTTIME("\n - Decrypt");

	/* Print Data Value / Pr[Y=1|X=x] for given test file using the dwData
	 * */
	SecureML::testProbAndYval("data/MIMIC.csv", dwData);

	/* Compute AUROC value
	 * */
	SecureML::testAUROC("data/MIMIC.csv", dwData);

	delete[] dwData;
	return 0;
}



