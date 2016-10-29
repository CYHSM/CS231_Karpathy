def permutation(string_in, string_prefix):
    if len(string_in) == 0:
        print(string_prefix)
    else:
        for i in range(len(string_in)):
            rem = string_in[:i] + string_in[i+1:]
            # print(rem)
            permutation(rem, string_prefix + string_in[i])
