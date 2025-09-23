def Try1( nums: list[int], target: int) -> list[int]:
        answer = []
        for index, value in enumerate(nums):
           subtracted_num = target - value
           if subtracted_num in nums:
                index2 = nums.index(subtracted_num)
                answer.append([index,index2])
                # answer.append(index2)
                # return (answer)
        return (answer)

###############################

def Solution(nums,target):
        seen = {}
        for i, v in enumerate(nums):
            remaining = target - v
            if remaining in seen:
               return [seen[remaining], i]
            seen[v] = i
        return []

##########################

nums = [3,2,0,1,4]
target = 5
A= Try1(nums,target)
print(A)
# B= Solution(nums,target)
# print(B)

# print('end')