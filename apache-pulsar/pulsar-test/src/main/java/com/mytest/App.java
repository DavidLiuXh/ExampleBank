package com.mytest;

import org.apache.pulsar.client.api.PulsarClient;
import org.apache.pulsar.client.api.Producer;
import org.apache.pulsar.client.api.PulsarClientException;

/**
 * Hello world!
 *
 */
public class App {
  private static String SERVICE_URI = "pulsar://10.0.2.15:6050,10.0.2.15:6051,10.0.2.15:6052";

  private static void SimpleProduceMsg() {
    try {
      PulsarClient client = PulsarClient.builder()
        .serviceUrl(SERVICE_URI)
        .build();

      Producer<byte[]> producer = client.newProducer()
        .topic("my-pulsar-1")
        .create();

      producer.send("My message".getBytes());

      producer.close();
      client.close();
    } catch (PulsarClientException e) {
      e.printStackTrace();
    }
  }

  public static void main( String[] args ) {
    SimpleProduceMsg();
  }
}
