#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Add_Two_Numbers.py
@time: 2018/1/25 19:22
@desc: Leetcode第二题
"""


'''
Example:
  Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
  Output: 7 -> 0 -> 8
  Explanation: 342 + 465 = 807.
'''
'''
  思路:
    先构建一个空的头结点不动，然后尾节点从头结点开始向后不断生成薪的节点，遍历两条链的公共部分
    每次相加相应位数字和进位，分配到结果的链表中，公共部分遍历完后再确定长的链表剩余的部分，同样的方式遍历完。
'''

class Solution(object):

    def addTwoNumbers(self, l1, l2):
        '''
        :param l1:
        :param l2:
        :return:
        '''



if __name__ == '__main__':
    l1 = [2, 4, 3]
    l2 = [5, 6, 4]