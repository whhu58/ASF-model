# Import socket module
from socket import *    

# Create a TCP server socket
#(AF_INET is used for IPv4 protocols)
#(SOCK_STREAM is used for TCP)

# Fill in start
serverPort = 6789
#serverip = '192.168.1.6'
serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind(("", serverPort))
serverSocket.listen(1)
print(serverSocket.getsockname())
# Fill in end 

# Server should be up and running and listening to the incoming connections
while True:
	print('Ready to serve...')
	
	# Set up a new connection from the client
	connectionSocket, addr = serverSocket.accept()#Fill in start              #Fill in end
	
	# If an exception occurs during the execution of try clause
	# the rest of the clause is skipped
	# If the exception type matches the word after except
	# the except clause is executed
	try:
		# Receives the request message from the client
		message = connectionSocket.recv(1024)
		print ("Received From Client: ", message)#Fill in start           #Fill in en
		# Extract the path of the requested object from the message
		# The path is the second part of HTTP header, identified by [1]
		filename = message.split()[1]	
		filetype = filename.split(b'.')[-1]
		if filename.endswith(b'jpg'):
			f = open(filename[1:], 'rb')
			print(filename)
            # set the content type for image/jpeg
			content_type = "image/jpeg"
		else:
			f = open(filename[1:], 'r')
            # set the content type for text/html
			content_type = "text/html"
			
		# Because the extracted path of the HTTP request includes 
		# a character '\', we read the path from the second character 

		# Store the entire contenet of the requested file in a temporary buffer
		outputdata = f.read()#Fill in start         #Fill in end

		# Send the HTTP response header line to the connection socket
		# Fill in start
		connectionSocket.send(bytes("HTTP/1.1 200 OK\r\n","UTF-8"))
		connectionSocket.send(bytes("Content-Type: {}\r\n".format(content_type),"UTF-8"))
		connectionSocket.send(bytes("\r\n","UTF-8"))
        # Fill in end
 
		# Send the content of the requested file to the connection socket
		if content_type == "text/html":
            # send as string for text/html
			connectionSocket.send(outputdata.encode())
		else:
            # send as bytes for image/jpeg
			connectionSocket.send(outputdata)
			#connectionSocket.send("\r\n".encode())
		print ("Success! File sent!")
		
		# Close the client connection socket
		connectionSocket.close()

	except IOError:
		# Send HTTP response message for file not found
		# Fill in start
		connectionSocket.send(bytes("HTTP/1.1 404 Not Found\r\n\r\n","UTF-8"))
		connectionSocket.send(bytes("<html><head></head><body><h1>404 Not Found</h1></body></html>\r\n","UTF-8"))
        # Fill in end
        
		# Close the client connection socket
		# Fill in start
		connectionSocket.close()
		serverSocket.close()
        # Fill in end

serverSocket.close()  

