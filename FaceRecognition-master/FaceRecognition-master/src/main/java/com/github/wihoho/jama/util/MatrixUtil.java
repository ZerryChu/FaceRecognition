package com.github.wihoho.jama.util;

import java.io.File;
import java.io.IOException;

import com.github.wihoho.jama.Matrix;
import com.github.wihoho.training.FileManager;

public class MatrixUtil {
    ClassLoader classLoader = getClass().getClassLoader();

	public Matrix convertToMatrix(String fileAddress) throws IOException {
        File file = new File(classLoader.getResource(fileAddress).getFile());
        return vectorize(FileManager.convertPGMtoMatrix(file.getAbsolutePath()));
    }

    //Convert a m by n matrix into a m*n by 1 matrix
    Matrix vectorize(Matrix input) {
        int m = input.getRowDimension();
        int n = input.getColumnDimension();

        Matrix result = new Matrix(m * n, 1);
        for (int p = 0; p < n; p++) {
            for (int q = 0; q < m; q++) {
                result.set(p * m + q, 0, input.get(q, p));
            }
        }
        return result;
    }
}
