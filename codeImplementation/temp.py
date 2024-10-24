def countNoOfFriends(N, M, K, tickets):
    # Convert tickets from set to list for proper indexing
    tickets = list(tickets)
    
    # John's ticket is the last one in the list
    john_ticket = tickets[-1]

    count = 0
    # Compare John's ticket with each of his friends' tickets
    for i in range(N):
        friend_ticket = tickets[i]
        # Count the number of differing bits between John's and friend's tickets
        if bin(john_ticket ^ friend_ticket).count('1') <= K:
            count += 1

    return count

# Test the function
print(countNoOfFriends(4, 5, 1, {1, 2, 4,8,16}))
