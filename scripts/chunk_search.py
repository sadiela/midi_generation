
intro = "ababcde"
verse = "abfgfggf"
chorus = "hihihijjjhi"
bridge = "jjkkjikakj"
outro = "abbabbcde"

parts = {
    'intro': intro,
    'verse': verse,
    'chorus': chorus,
    'bridge': bridge,
    'outro': outro
}

song1 = intro + verse + chorus + verse + chorus + bridge + chorus + outro


def LCSubStr(X, Y):
 
    # Create a table to store lengths of
    # longest common suffixes of substrings.
    # Note that LCSuff[i][j] contains the
    # length of longest common suffix of
    # X[0...i-1] and Y[0...j-1]. The first
    # row and first column entries have no
    # logical meaning, they are used only
    # for simplicity of the program.
    m = len(X)
    n = len(Y)
 
    LCSuff = [[0 for i in range(n + 1)]
                 for j in range(m + 1)]
 
    # To store length of the
    # longest common substring
    length = 0
 
    # To store the index of the cell
    # which contains the maximum value.
    # This cell's index helps in building
    # up the longest common substring
    # from right to left.
    row, col = 0, 0
 
    # Following steps build LCSuff[m+1][n+1]
    # in bottom up fashion.
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                LCSuff[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1
                if length < LCSuff[i][j]:
                    length = LCSuff[i][j]
                    row = i
                    col = j
            else:
                LCSuff[i][j] = 0
 
    # if true, then no common substring exists
    if length == 0:
        print("No Common Substring")
        return
 
    # allocate space for the longest
    # common substring
    resultStr = ['0'] * length
 
    # traverse up diagonally form the
    # (row, col) cell until LCSuff[row][col] != 0
    while LCSuff[row][col] != 0:
        length -= 1
        resultStr[length] = X[row - 1] # or Y[col-1]
 
        # move diagonally up to previous cell
        row -= 1
        col -= 1
 
    # required longest common substring
    return ''.join(resultStr)

def longestCommonPrefix(strs):
      """
      :type strs: List[str]
      :rtype: str
      """
      if len(strs) == 0:
         return ""
      current = strs[0]
      for i in range(1,len(strs)):
         temp = ""
         if len(current) == 0:
            break
         for j in range(len(strs[i])):
            if j<len(current) and current[j] == strs[i][j]:
               temp+=current[j]
            else:
               break
         current = temp
      return current

def kmer_finder(k=5):
    # find repeating subsequences of length k
    return "Empty"

cutoff = len(song1)//2
str_list = [song1[:cutoff], song1[cutoff:]]
print(str_list)

lcp = LCSubStr(song1[:cutoff], song1[cutoff:])
print(song1)
print(lcp)

for k in parts.keys():
    if parts[k] in lcp or lcp in parts[k]:
        print(k)