# Problem #27
# Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. 
# The order of the elements may be changed. Then return the number of elements in nums which are not equal to val.
# Consider the number of elements in nums which are not equal to val be k, to get accepted, you need to do the following things:
# Change the array nums such that the first k elements of nums contain the elements which are not equal to val. 
# The remaining elements of nums are not important as well as the size of nums.
# Return k.
# Example 2:
#    Input: nums = [0,1,2,2,3,0,4,2], val = 2
#    Output: 5, nums = [0,1,4,0,3,_,_,_]

#########################################

class Solution:
    def removeElement(self, nums:list[int], val:int) ->int:
        check = 0
        while check < len(nums):
            if nums[check] != val:
                check +=1
            else:
                nums.pop(check)
        print(f'>>> k={len(nums)}, {nums}')
        return()
    
    # def removeElement_2(self, nums:list[int], val:int) ->int:
    #     index = 0
    #     for i in range(len(nums)):
    #         if nums[i] != val:
    #             nums[index] = nums[i]
    #             index += 1
    #     print(nums)
    #     return index 

solution = Solution()
result = solution.removeElement(nums=[2,2,1,2,2,3,0,4,2],val=2)