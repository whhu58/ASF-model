from socket import *
import os

cache = {}  # dictionary to store cached responses

serverPort = 8888
serverip = '0.0.0.0'
serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind((serverip, serverPort))
serverSocket.listen(5)

webHost = '127.0.0.1'
webPort = 80

print("The proxy server is ready to receive...")

while True:
    print('Ready to serve...')

    connectionSocket, addr = serverSocket.accept()

    try:
        message = connectionSocket.recv(1024)
        print("Received From Client: ", message)

        splited = message.decode().split(" ")[1].split("/")

        if len(splited) <= 2:
            filePart = splited[1]
        else:
            hostPart = splited[1]
            filePart = splited[2]

            webHost = hostPart.split(':')[0]
            webPort = hostPart.split(':')[1]

        filename = filePart.split(".")[0]
        filetype = filePart.split(".")[1]

        splitMessage = message.decode().split(" ")
        splitMessage[1] = "/" + filePart
        newMessage = bytes(" ".join(splitMessage), "UTF-8")

        if filename.startswith("http://") or filename.startswith("https://"):
            # extract path component from URL
            path = filename.split("/")[-1]
            filename = "/" + path

        print(filetype)

        if filename in cache:
            # if the response is in the cache, send it back to the client
            print("Cached response found!")
            connectionSocket.send(cache[filename])
        else:
            # otherwise, forward the request to the web server
            print("Forwarding request to web server...")
            webServerName = webHost
            webServerPort = int(webPort)
            clientSocket = socket(AF_INET, SOCK_STREAM)
            clientSocket.connect((webServerName, webServerPort))

            clientSocket.send(newMessage)
            # receive the response from the web server
            response = b""
            while True:
                data = clientSocket.recv(1024)
                if not data:
                    break
                response += data
            # add the response to the cache and send it back to the client
            cache[filename] = response
            connectionSocket.send(response)

            # close the connection to the web server
            clientSocket.close()

        connectionSocket.close()

    except IOError:
        print("File not found!")
        connectionSocket.send(bytes("HTTP/1.1 404 Not Found\r\n\r\n", "UTF-8"))
        connectionSocket.send(bytes("<html><head></head><body><h1>404 Not Found</h1></body></html>\r\n", "UTF-8"))
        connectionSocket.close()

serverSocket.close()
