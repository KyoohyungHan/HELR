#include <iostream>
#include <stdlib.h>
#include <chrono>

#include <NTL/BasicThreadPool.h>

#include "HEAAN.h"

#include "Params.h"
#include "ML.h"
#include "functions.h"

chrono::high_resolution_clock::time_point t1, t2;

#define START() t1 = chrono::high_resolution_clock::now();
#define END() t2 = chrono::high_resolution_clock::now();
#define PRINTTIME(msg) cout << msg << " time = " << (double)chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() / 1000 << " seconds" << endl;

using namespace std;
using namespace NTL;

void test(string file, string file_test, bool isFirst, long numThread, bool isEncrypted) {

	/* Construct BasicThreadPool and set the number of thread
	 * */
	BasicThreadPool pool(numThread);

	/* Read Data file and shuffle and normalize it
	 * */
	long factorNum, sampleNum;
	double** zData = SecureML::zDataFromFile(file, factorNum, sampleNum, isFirst);

	SecureML::shuffleZData(zData, factorNum, sampleNum);
	SecureML::normalizeZData(zData, factorNum, sampleNum);
	
	/* Params(long factorNum, long sampleNum, long iterNum, double alpha, long numThread)
	 * alpha : learning rate (0.001 to 1.0 values are used in general
	 * iterNum does not need to be large because we use mini-batch technique + optimized gradient decent
	 */
	SecureML::Params params(factorNum, sampleNum, 32, 1.0, pool.NumThreads());
	params.path_to_file = file;
	params.path_to_test_file = file_test;
	params.isfirst = isFirst; ///< Y is at the first column of the data?

	/* Key Generation for HEAAN library
	 * If params.iterNum is larger than params.iterNumPerBoot, this will generate public key for bootstrapping
	 * */
	START();
	Ring ring(params.logN, params.logQBoot);
	SecretKey sk(ring);
	Scheme scheme(sk, ring, true); ///< save all public keys in /serkey folder
	if(isEncrypted) {
		scheme.addLeftRotKeys(sk);
		scheme.addRightRotKeys(sk);
		scheme.addBootKey(sk, params.bBits, params.logq + params.logI);
	}
	END(); PRINTTIME("\n - KeyGen");

	/* Simple Constructor for secureML class
	 * This will just pass some address of scheme, params, and pool
	 * */
	SecureML::ML secureML(scheme, params, pool, sk);

	/* Encrypt zData using public key (this part can be faster in symmetric key version)
	 * To prepare for large number of data case, encryption of zData will be saved in encData/ folder with .txt files
	 * */
	START();
	if(isEncrypted) secureML.EncryptzData(zData, factorNum, sampleNum);
	END(); PRINTTIME(" - Encrypt");

	/* Training using zData which are saved as txt file
	 * */
	START();
	double* pwData = new double[params.factorNum]();
	Ciphertext* encWData = new Ciphertext[params.cnum];
	if(isEncrypted) secureML.Training(encWData, factorNum, sampleNum, pwData, zData);
	else secureML.plainTraining(pwData, zData, factorNum, sampleNum);
	END(); PRINTTIME("\n - Training");

	if(isEncrypted) {
		/* Decrypt encWData using secretkey sk and save dwData in dwData.csv
		 * */
		secureML.DecryptwDataAndSave("dwData.csv", encWData, factorNum);
	}
	delete[] pwData;
	delete[] encWData;
}

int main(int argc, char* argv[]) {

	string file1("../data/" + string(argv[1])); ///< file name for training
	string file2("../data/" + string(argv[2])); ///< file name for testing
	bool isFirst = atoi(argv[3]); //> the target value is at the first column or not?
	bool isEncrypted = atoi(argv[4]); ///> logistic regression for encrypted state?
	long numThread = atoi(argv[5]); ///> number of threads in multi-threading

	if(isEncrypted) {
		cout << "HELR Test with thread " << numThread << endl;
		cout << "Training Data = " << file1 << endl;
		cout << "Testing Data = " << file2 << endl;
	} else {
		cout << "LR Test with thread " << numThread << endl;
		cout << "Training Data = " << file1 << endl;
		cout << "Testing Data = " << file2 << endl;
	}
	test(file1, file2, isFirst, numThread, isEncrypted);
	return 0;
}



