# dwtest-varification



## Goal

    Verify the correctnes and  compare the power of dwtest in R and in Python
    ref: https://github.com/dima-quant/dwtest
    hackMD: https://hackmd.io/FykfBMo8RkKg7_ySSRnipw?view
## Guildline
Put all .py, .ipynb, .Rmd in same level of directory:

    - First, you can either use generating_csv.py or allprocess.ipynb to generate .csvs.
    - Second, you can use varification.py or allprocess.ipynb to conduct dwtest in python.
    - Third, you can use testdwtest.Rmd to conduct dwtest in python
    - Forth, you can use experiment.ipynb or you own code to conduct experiment on type I,II error
## error rate comparison
For rho equal to 0:

    - We calculate Type I error rate
    
For rho not equal to 0:

    - For right-tail and two-tail test:
    
        We calculate Type II error rate
        
    - For left-tail test:
    
        We calculate Type I error rate
