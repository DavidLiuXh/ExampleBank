package com.mytest;

import org.apache.pulsar.client.api.PulsarClient;
import org.apache.pulsar.client.api.Producer;
import org.apache.pulsar.client.api.Consumer;
import org.apache.pulsar.client.api.Message;
import org.apache.pulsar.client.api.PulsarClientException;

/**
 * Hello world!
 *
 */
public class App {
  private static String SERVICE_URI = "pulsar://10.0.2.15:6650,10.0.2.15:6651,10.0.2.15:6652";

  private static boolean kStop = false;

  private static void SimpleProduceMsg(String topic) {
    try {
      PulsarClient client = PulsarClient.builder()
        .serviceUrl(SERVICE_URI)
        .build();

      Producer<byte[]> producer = client.newProducer()
        .topic(topic)
        .create();

      producer.send("My message".getBytes());

      producer.close();
      client.close();
    } catch (PulsarClientException e) {
      e.printStackTrace();
    }
  }

  private static void SimpleConsumeMsg(String topic) {
    try {
      PulsarClient client = PulsarClient.builder()
        .serviceUrl(SERVICE_URI)
        .build();

      Consumer consumer = client.newConsumer()
        .topic(topic)
        .subscriptionName("my-subscription")
        .subscribe();

      int i = 0;
      while(++i < 2) {
        Message msg = consumer.receive();

        try {
          System.out.printf("Message received: %s\n", new String(msg.getData()));
          
          consumer.acknowledge(msg);
        } catch (Exception e) {
          // Message failed to process, redeliver later
          consumer.negativeAcknowledge(msg);
        }
      }

      consumer.close();
      client.close();
    } catch (PulsarClientException e) {
      e.printStackTrace();
    }
  }

  private static class SimpleProduceThread extends Thread {
    private String topic;

    public SimpleProduceThread(String topic) {
      this.topic = topic;
    }

    public void run() {
      try {
        PulsarClient client = PulsarClient.builder()
          .serviceUrl(SERVICE_URI)
          .build();

        Producer<byte[]> producer = client.newProducer()
          .topic(topic)
          .create();

        int i = 0;
        while (!kStop) {
          producer.send(String.valueOf(++i).getBytes());
        }

        producer.close();
        client.close();
      } catch (PulsarClientException e) {
        e.printStackTrace();
      }
    }
  }

  private static class SimpleConsumeThread extends Thread {
    private String topic;

    public SimpleConsumeThread(String topic) {
      this.topic = topic;
    }

    public void run() {
      try {
        PulsarClient client = PulsarClient.builder()
          .serviceUrl(SERVICE_URI)
          .build();

        Consumer consumer = client.newConsumer()
          .topic(topic)
          .subscriptionName("my-subscription")
          .subscribe();

        while(!kStop) {
          Message msg = consumer.receive();

          try {
            System.out.printf("Message received: %s\n", new String(msg.getData()));

            consumer.acknowledge(msg);
          } catch (Exception e) {
            // Message failed to process, redeliver later
            consumer.negativeAcknowledge(msg);
          }
        }

        consumer.close();
        client.close();
      } catch (PulsarClientException e) {
        e.printStackTrace();
      }
    }
  }

  private static void SimpleProduceAndConsume(String topic) {
    SimpleProduceThread producer = new SimpleProduceThread(topic);
    producer.start();

    SimpleConsumeThread consumer = new SimpleConsumeThread(topic);
    consumer.start();

    try {
      producer.join();
      consumer.join();
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
  }

  private static class ExitHandler extends Thread {
    public ExitHandler() {
      super("Exit Handler");
    }
    
    public void run() {
      System.out.println("Set exit");
      kStop = true;
    }
  }

  public static void main( String[] args ) {
    Runtime.getRuntime().addShutdownHook(new ExitHandler());

    //SimpleProduceMsg("my-pulsar-1");
    //SimpleConsumeMsg("my-pulsar-1");
    SimpleProduceAndConsume("my-pulsar-1");
  }
}
