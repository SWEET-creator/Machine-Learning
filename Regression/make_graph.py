import matplotlib.pyplot as plt


label = ["線形",
        "リッジ",
        "多項式回帰（2次）",
        "ベイジアンリッジ",
        "SVM(kernel = “liner” C=1.0, epsilon=0.1)",
        "SVM(kernel = “poly” C=1.0, epsilon=0.1)"]

x = [1,2]
values = ["MSE(train data)", "MSE(test data)"]

y1 = [0.4217886308867676,
0.4217888572263998,
0.36989621297807435,
0.3940383259334631,
0.391188281242315,
0.391188281242315,
]

y2 = [0.408848171145515,
0.40883233250147855,
0.47074166942638884,
0.39321451890211767,
0.4092321070586411,
0.4092321070586411,
]
plt.ylim(0.35,0.5)
plt.plot(x,[y1, y2])
plt.xticks(x,values)
plt.legend(label, prop={"family":"Hiragino Mincho ProN"})
plt.savefig("MSE.jpg")