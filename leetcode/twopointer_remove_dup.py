# Problem #26

class Solution:
    def removeDuplicates(self, nums: list[int]) -> int:
        pos = 0
        for check in range(1,len(nums)):
            if nums[check] != nums[pos]:
                nums[pos+1] = nums[check]
                pos += 1
        return(pos+1,nums)
    
solution = Solution()
pos, result = solution.removeDuplicates(nums=[0,0,1,1,1,2,2,3,3,4,4,6,7])
print(f'k={pos}, {result}')
