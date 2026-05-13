import pika
import json


class RabbitMQClient:

    def __init__(self,
                 host,
                 username,
                 password):

        credentials = pika.PlainCredentials(
            username,
            password
        )

        parameters = pika.ConnectionParameters(
            host=host,
            credentials=credentials
        )

        self.connection = pika.BlockingConnection(
            parameters
        )

        self.channel = self.connection.channel()

    def declare_queue(self,
                      queue_name):

        self.channel.queue_declare(
            queue=queue_name
        )

    def publish_json(self,
                     queue_name,
                     data):

        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=json.dumps(data)
        )

    def publish_binary(self,
                       queue_name,
                       data):

        self.channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=data
        )

    def consume(self,
                queue_name,
                callback):

        self.channel.basic_consume(
            queue=queue_name,
            on_message_callback=callback,
            auto_ack=True
        )

        self.channel.start_consuming()