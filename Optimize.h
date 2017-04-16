#pragma once
#include"Preprocessing.h"


VectorXd OptimizeGD(MatrixXd inputs, MatrixXd answers) {
	int size = inputs.cols();
	int ans = answers.cols();
	int nums = inputs.rows();
	//variables
	MatrixXd X = MatrixXd::Random(size, ans);

	//MatrixXd sqPart = (X.transpose()*inputs)*X;
	//need to check when to 
	double oldErr = 2000000;
	double newErr = 1000000;
	
	
	
	MatrixXd err;
	MatrixXd ones = VectorXd::Ones(inputs.rows(), X.cols());//ones vector
	MatrixXd dev = (inputs - ones*ones.transpose()*inputs) / nums;//deviation matrix
	MatrixXd varcov = dev.transpose()*dev / nums;//vatiance-covariance matrix
	
	double lr = 5*pow(10, -11);//learning rate
	while (true) {
		if (abs(oldErr - newErr) > 0.01) {//if learning had significant effect
			cout << "inside" << endl;
			MatrixXd lin = inputs*X;//linear part
			MatrixXd quad = ones*(X.transpose()*(varcov)*X);//quadratic part
			err = (0.5*quad + lin - answers);//error - purpose function
			
			cout << err.sum() << endl;
			oldErr = newErr;
			newErr = err.squaredNorm();
			double old = X.sum();
			//update
			X = X - lr*(((err.transpose()/nums)*((ones*((varcov*X)).transpose()) + inputs)).transpose()/nums + (X)/5);
			cout << "changed" << endl;
			cout << X.sum() - old << endl;
		}
		else {
			if (oldErr - newErr < 0) {//if jumped above minimum - decrease LR
				lr = lr / 10;
			}
			else {
				break;
			}
		}
	}
	
	//cout << X.transpose() << endl;
	cout << err.sum() << endl;
	return X;
}

void testGD(MatrixXd inputs, MatrixXd answers, MatrixXd vars) {//test
	int size = inputs.cols();
	int ans = answers.cols();
	int nums = inputs.rows();

	MatrixXd ones = VectorXd::Ones(inputs.rows(), vars.cols());
	MatrixXd dev = (inputs - ones*ones.transpose()*inputs) / nums;
	MatrixXd varcov = dev.transpose()*dev / nums;
	MatrixXd lin = inputs*vars;
	MatrixXd quad = ones*(vars.transpose()*(varcov)*vars);

	MatrixXd err = 0.5*quad + lin - answers;
	
	int T = 0;//true
	int F = 0;//false
	for (int i = 0; i < nums; i++) {
		if ((err.row(i)*err.row(i)).sum() <= 0.25) {
			T++;
		}
		else {
			F++;
		}
	}

	cout << "accuracy = " << T / (T + F) << endl;

	system("pause");
}

VectorXd Nesterov(MatrixXd inputs, MatrixXd answers) {
	int size = inputs.cols();
	int ans = answers.cols();
	int nums = inputs.rows();

	MatrixXd X = MatrixXd::Random(size, ans);

	//MatrixXd sqPart = (X.transpose()*inputs)*X;

	double oldErr = 2000000;
	double newErr = 1000000;



	MatrixXd err;
	MatrixXd ones = VectorXd::Ones(inputs.rows(), X.cols());
	MatrixXd dev = (inputs - ones*ones.transpose()*inputs) / nums;
	MatrixXd varcov = dev.transpose()*dev / nums;

	MatrixXd V = MatrixXd::Zero(X.rows(), X.cols());
	
	double lr = pow(10, -10);
	double gamma = 1 - lr;
	double eta = lr;
	while (true) {
		if (abs(oldErr - newErr) > 0.01) {
			cout << "inside" << endl;
			MatrixXd Vin = MatrixXd::Zero(X.rows(), X.cols());
			Vin = X - gamma*V;
			MatrixXd lin = inputs*Vin;
			MatrixXd quad = ones*(Vin.transpose()*(varcov)*Vin);
			err = (0.5*quad + lin - answers);
			MatrixXd grad = ((err.transpose()/nums)*((ones*((varcov*Vin)).transpose()) + inputs)).transpose() / nums;
			V = gamma*V + eta*grad;
			cout << err.squaredNorm() << endl;
			oldErr = newErr;
			newErr = err.sum();
			double old = X.sum();
			X = X - V;
			cout << "changed" << endl;
			cout << X.sum() - old << endl;
		}
		else {
			if (oldErr - newErr < 0) {
				lr = lr / 10;
			}
			else {
				break;
			}
		}
	}

	//cout << X.transpose() << endl;
	cout << err.sum() << endl;
	return X;
}