#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Merge_Two_Sorted_Lists.py 
@desc: 合并两个链表，从小到大排序
@time: 2018/02/03 
"""

'''
   Input: 1->2->4, 1->3->4
   Output: 1->1->2->3->4->4
'''

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """