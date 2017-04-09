package com.github.wihoho;

import com.github.wihoho.constant.FeatureType;
import com.github.wihoho.jama.Matrix;
import com.github.wihoho.training.*;
import com.google.common.base.Preconditions;

import java.util.ArrayList;
import java.util.Objects;

public class Trainer {
	public Metric metric;
	public FeatureType featureType;
	public FeatureExtraction featureExtraction;
	public int numberOfComponents;
	public int k; // k specifies the number of neighbour to consider

	public ArrayList<Matrix> trainingSet;
	public ArrayList<String> trainingLabels;

	public ArrayList<ProjectedTrainingMatrix> model;

	public void add(Matrix matrix, String label) {
		if (Objects.isNull(trainingSet)) {
			trainingSet = new ArrayList<>();
			trainingLabels = new ArrayList<>();
		}

		trainingSet.add(matrix);
		trainingLabels.add(label);
	}

	public void addFaceAfterTraining(Matrix matrix, String label) {
		featureExtraction.addFace(matrix, label);
	}

	public void train() throws Exception {
		Preconditions.checkNotNull(metric);
		Preconditions.checkNotNull(featureType);
		Preconditions.checkNotNull(numberOfComponents);
		Preconditions.checkNotNull(trainingSet);
		Preconditions.checkNotNull(trainingLabels);

		switch (featureType) {
		case PCA:
			featureExtraction = new PCA(trainingSet, trainingLabels, numberOfComponents);
			break;
		case LDA:
			featureExtraction = new LDA(trainingSet, trainingLabels, numberOfComponents);
			break;
		case LPP:
			featureExtraction = new LPP(trainingSet, trainingLabels, numberOfComponents);
			break;
		}

		model = featureExtraction.getProjectedTrainingSet();
	}

	public String recognize(Matrix matrix) {
		Matrix testCase = featureExtraction.getW().transpose().times(matrix.minus(featureExtraction.getMeanMatrix()));
		String result = KNN.assignLabel(model.toArray(new ProjectedTrainingMatrix[0]), testCase, k, metric);
		return result;
	}

}
