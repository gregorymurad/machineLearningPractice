def find_substrings_in_list(list1, list2):
    print('list1: ', list1)
    print('list2: ', list2)
    res = [True if any(j in i for j in list2) else False for i in list1] # O(nˆ2)
    print(res)
    r = [x for x,y in zip(list1, res) if y == True] # O(n)
    return r

if __name__ == '__main__':
    list_a = ['abc', 'bcd', 'cde', 'def', ''] #string
    list_b = ['a', 'e'] #substring
    r = find_substrings_in_list(list_a, list_b)
    print('r: ', r)

