#Surabhi S Nath
#2016271

#Import statements
import math
import sklearn
import numpy
import sys



def main(argv):

    #Files taken as arguement from command line
    infile1 = argv[0]   #Neg file
    infile2 = argv[1]   #Pos file
    outfile1 = argv[2]  #Answer to Question1, Neg
    outfile2 = argv[3]  #Answer to Question1, Pos
    outfile3 = argv[4]  #Answer to Question2, Neg
    outfile4 = argv[5]  #Answer to Question2, Pos

    #Dictionary mapping amino acid to array index
    dict = {'G':0, 'A':1, 'S':2, 'T':3, 'C':4, 'V':5, 'L':6, 'I':7, 'M':8, 'P':9, 'F':10, 'Y':11, 'W':12, 'D':13, 'E':14, 'N':15, 'Q':16, 'H':17, 'K':18, 'R':19}

    #Initialize array representing % amino acide composition
    globalarrneg = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    globalarrneg = numpy.array(globalarrneg)
    globalarrpos = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    globalarrpos = numpy.array(globalarrpos)
    sumlenlineneg = 0
    sumlenlinepos = 0

    #Open the output files where results will be written
    ans11 = open(outfile1,'w')
    ans12 = open(outfile2,'w')
    ans21 = open(outfile3,'w')
    ans22 = open(outfile4,'w')

    ans11.write("For Negative Cases\n\n")
    ans11.write("Amino Acid Sequence:\n"+"[G, A, S, T, C, V, L, I, M, P, F, Y, W, D, E, N, Q, H, K, R]\n\n")
    cntneg = 0

    #Calculate % composititon for each sequence
    with open(infile1) as f:
        for line in f:
            cntneg = cntneg + 1
            arr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            line = line[:-1]

            sumlenlineneg = sumlenlineneg + len(line)
            
            #Increase count of index according to amino acid
            for i in range(0,len(line)):
                arr[dict.get(line[i])] = arr[dict.get(line[i])] + 1

            #Find %s
            arr = numpy.array(arr)
            globalarrneg = globalarrneg + arr 
            arr = arr/len(line)*100

            #Write % composition to file
            ans11.write("Peptide Sequence #"+str(cntneg)+" Compositions:\n"+numpy.array2string(arr)+"\n\n")

    #Write total number of sequences to file
    ans11.write("Total Number of Sequences = "+str(cntneg))
    #Close file
    ans11.close()

    ans12.write("For Positive Cases\n\n")
    ans12.write("Amino Acid Sequence:\n"+"[G, A, S, T, C, V, L, I, M, P, F, Y, W, D, E, N, Q, H, K, R]\n\n")
    cntpos = 0

    #Calculate % composititon for each sequence
    with open(infile2) as f:
        for line in f:
            cntpos = cntpos + 1
            arr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            line = line[:-1]

            sumlenlinepos = sumlenlinepos + len(line)

            #Increase count of index according to amino acid
            for i in range(0,len(line)):
                arr[dict.get(line[i])] = arr[dict.get(line[i])] + 1

            #Find %s
            arr = numpy.array(arr)
            globalarrpos = globalarrpos + arr
            arr = arr/len(line)*100

            #Write % composition to file
            ans12.write("Peptide Sequence #"+str(cntpos)+" Compositions:\n"+numpy.array2string(arr)+"\n\n")

    #Write total number of sequences to file
    ans12.write("Total Number of Sequences = "+str(cntpos))
    #Close file
    ans12.close()

    #--------------------------------------------------------------------------------------------------
    #Part b

    #Write mean amino acid composition for entire file of negative and positive to ans21 and ans22
    ans2neg = globalarrneg/sumlenlineneg*100
    ans2pos = globalarrpos/sumlenlinepos*100
    ans21.write("Mean Amino Acid Composition for entire input file of Negative Sequences:\n\n")
    ans21.write("Amino Acid Sequence:\n"+"[G, A, S, T, C, V, L, I, M, P, F, Y, W, D, E, N, Q, H, K, R]\n\n")
    ans21.write(numpy.array2string(ans2neg))
    ans22.write("Mean Amino Acid Composition for entire input file of Positive Sequences:\n\n")
    ans22.write("Amino Acid Sequence:\n"+"[G, A, S, T, C, V, L, I, M, P, F, Y, W, D, E, N, Q, H, K, R]\n\n")
    ans22.write(numpy.array2string(ans2pos))
    
    #Close files
    ans21.close()
    ans22.close()


if __name__ == "__main__":
   main(sys.argv[1:])
