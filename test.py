import part1 as p1
import matplotlib.pyplot as plt
import csv

cipherfunc = [];      
ciphertext = [];
plaintext = [];

with open('./cipher_function.csv') as csvfile:
    filereader = csv.reader(csvfile)
    for i, row in enumerate(filereader):
        cipherfunc  = row;
with open('./ciphertext.txt') as f:
    ciphertext = f.read();
    
with open('./plaintext.txt') as f:
    plaintext = f.read();

# REMOVE PLAINTEXT BEFORE SUBMITTION
f = p1.decode(ciphertext, './converted_plain_text.txt')
print "Infer: \t", f
print "True : \t", cipherfunc 

accuracy = p1.calc_accuracy(f, ciphertext, plaintext)

print "Accuracy:", accuracy

"""
plt.figure(1)
plt.title('Accuracy rate @ length = ' + str(length))
plt.xlabel('Sample t')
plt.ylabel('Correct symbols / total symbols')
plt.plot(accuracy_ratio)

plt.figure(2)
plt.subplot(221)
plt.title('Log-likelihood of accepted state @ length = ' + str(length))
plt.xlabel('Sample t')
plt.ylabel('Log-likelihood p(y | f)')
plt.plot(log_likelihood)

plt.subplot(222)
plt.title('Symbol Log-likelihood @ length = ' + str(length))
plt.xlabel('Sample t')
plt.ylabel('Symbol Log-likelihood p(y | f)/N')
plt.plot(symbol_log_likelihood)

plt.subplot(223)
plt.title('Acceptence rate for T = 100 @ length = ' + str(length))
plt.xlabel('Sample t')
plt.ylabel('Accepted samples / total samples')
plt.plot(accept_ratio)

plt.subplot(224)
plt.title('Accuracy rate @ length = ' + str(length))
plt.xlabel('Sample t')
plt.ylabel('Correct symbols / total symbols')
plt.plot(accuracy_ratio)

plt.show()
"""
