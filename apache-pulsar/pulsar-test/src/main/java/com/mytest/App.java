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
  private static String SERVICE_URI = "pulsar://10.173.229.17:6650,10.173.220.191:6650,10.173.220.190:6650";

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
    private long msgCount;

    public SimpleProduceThread(String topic, long msgCount) {
      this.topic = topic;
      this.msgCount = msgCount;
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
        if (msgCount > 0) {
          for (long j = 0; j < msgCount; ++j) {
            producer.send(String.valueOf(j).getBytes());
          }
        } else {
          while (!kStop) {
            producer.send(String.valueOf(++i).getBytes());
          }
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

  private static void SimpleProduceAndConsume(String topic,
      long msgCount,
      boolean notConsume,
      boolean notProduce) {
    SimpleProduceThread producer = new SimpleProduceThread(topic, msgCount);
    producer.start();

    SimpleConsumeThread consumer = null;
    if (!notConsume) {
      consumer = new SimpleConsumeThread(topic);
      consumer.start();
    }

    try {
      producer.join();

      if (!notConsume) {
        consumer.join();
      }
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
    System.out.printf("topic:%s\n", args[0]);

    Runtime.getRuntime().addShutdownHook(new ExitHandler());

    //SimpleProduceMsg("my-pulsar-1");
   // SimpleConsumeMsg(args[0]);
    SimpleProduceAndConsume(args[0], 10, false, false);
  }
}
