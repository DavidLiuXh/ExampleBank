package com.mytest;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.Condition;
import java.util.List;
import java.util.ArrayList;
import java.util.Iterator;

import org.apache.pulsar.client.api.PulsarClient;
import org.apache.pulsar.client.api.Producer;
import org.apache.pulsar.client.api.Consumer;
import org.apache.pulsar.client.api.Message;
import org.apache.pulsar.client.api.PulsarClientException;
import org.apache.pulsar.client.api.PulsarClientException.ProducerBlockedQuotaExceededException;

/**
 * Hello world!
 *
 */
public class App {
  private static String SERVICE_URI = "pulsar://10.173.229.17:6650,10.173.220.191:6650,10.173.220.190:6650";
  //private static String SERVICE_URI = "pulsar://10.0.2.15:6650,10.0.2.15:6651,10.0.2.15:6652";

  private static boolean kStop = false;

  private static Lock produceAndConsumeLock = new ReentrantLock();
  private static Condition produceAndConsumeCondition = produceAndConsumeLock.newCondition(); 

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

  private static void SimpleConsumeMsg(String topic,
      boolean ack) {
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

          if (ack) {
            consumer.acknowledge(msg);
          }
        } catch (Exception e) {
          // Message failed to process, redeliver later
          if (ack) {
            consumer.negativeAcknowledge(msg);
          }
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

        try {
          produceAndConsumeLock.lock();
          produceAndConsumeCondition.signal();
        } finally {
          produceAndConsumeLock.unlock();
        }

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

  // test producer is together with cosumer
  private static void SimpleProduceAndConsume(String topic,
      long msgCount,
      boolean notConsume,
      boolean notProduce) {
    SimpleConsumeThread consumer = null;
    if (!notConsume) {
      consumer = new SimpleConsumeThread(topic);
      consumer.start();

      try {
        produceAndConsumeLock.lock();
        produceAndConsumeCondition.await();
      } catch (InterruptedException e) {
        e.printStackTrace();
      } finally {
        produceAndConsumeLock.unlock();
      }
    }

    SimpleProduceThread producer = new SimpleProduceThread(topic, msgCount);
    producer.start();

    try {
      producer.join();

      if (!notConsume) {
        consumer.join();
      }
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
  }

  private static void FirstProduceThanConsume(String topic,
      long msgCount) {
    SimpleProduceThread producer = new SimpleProduceThread(topic, msgCount);
    producer.start();

    try {
      Thread.sleep(1000);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }

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

  // test backlog policy
  // ./pulsar-admin namespaces set-backlog-quota --limit 100000 --policy producer_exception my-tenants-1/my-namespace-1
  // http://pulsar.apache.org/docs/en/admin-api-namespaces/
  // 在Create Producer和 send时都可能抛出ProducerBlockedQuotaExceededException
  // send中抛出时，如果调整吧 limit，send可以恢复，不用重启producer
  private static void ConcurrentProduce(String topic,
      int concurrent,
      int msgCount,
      boolean notConsume) {
    List<Thread> threadList = new ArrayList(2);

    Thread consume = null;
    if (!notConsume) {
      consume = new Thread(() -> {
        SimpleConsumeMsg(topic, false);
      });
      consume.start();
    }

    for (int i = 0; i < concurrent; ++i) {
      Thread t = new Thread(() -> {
        try {
          PulsarClient client = PulsarClient.builder()
            .serviceUrl(SERVICE_URI)
            .build();

          Producer<byte[]> producer = null; 
          while (!kStop) {
            try {
              producer = client.newProducer()
                .topic(topic)
                .create();
              break;
            } catch (ProducerBlockedQuotaExceededException e) {
              try {
                Thread.sleep(1000);
              } catch (InterruptedException ee) {
                ee.printStackTrace();
              }

              e.printStackTrace();
            }
          }

          if (producer != null) {
            if (msgCount > 0) {
              for (long j = 0; j < msgCount; ++j) {
                try {
                  producer.send(String.valueOf(j).getBytes());
                } catch (PulsarClientException.ProducerBlockedQuotaExceededException e) {
                  try {
                    Thread.sleep(1000);
                  } catch (InterruptedException ee) {
                    ee.printStackTrace();
                  }
                  e.printStackTrace();
                } catch (PulsarClientException e) {
                  e.printStackTrace();
                }
              }
            } else {
              int k = 0;
              while (!kStop) {
                producer.send(String.valueOf(++k).getBytes());
              }
            }
          }

          producer.close();
          client.close();
        } catch (PulsarClientException.ProducerBlockedQuotaExceededException e) {
          e.printStackTrace();
        } catch (PulsarClientException e) {
          e.printStackTrace();
        }
      });

      threadList.add(t);

      t.start();
    }

    try {
      Iterator<Thread> it = threadList.iterator();
      while (it.hasNext()) {
        Thread t = it.next();
        t.join();
      }

      if (!notConsume) {
        consume.join();
      }
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
  }

  public static void main( String[] args ) {
    System.out.printf("topic:%s\n", args[0]);

    Runtime.getRuntime().addShutdownHook(new ExitHandler());

    //SimpleProduceMsg("my-pulsar-1");
    //SimpleConsumeMsg(args[0], true);
    //SimpleProduceAndConsume(args[0], 10, false, false);
    //FirstProduceThanConsume(args[0], 10);

    ConcurrentProduce(args[0], 10, 10000, false);
  }
}
