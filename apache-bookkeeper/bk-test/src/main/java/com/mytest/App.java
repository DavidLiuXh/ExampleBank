package com.mytest;

import java.io.IOException;
import java.util.Iterator;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;

import org.apache.bookkeeper.conf.ClientConfiguration;
import org.apache.bookkeeper.client.BookKeeper;
import org.apache.bookkeeper.client.api.LedgerEntry;
import org.apache.bookkeeper.client.api.WriteHandle;
import org.apache.bookkeeper.client.api.ReadHandle;
import org.apache.bookkeeper.client.api.LedgerEntries;
import org.apache.bookkeeper.client.api.DigestType;
import org.apache.bookkeeper.client.api.BKException;
import org.apache.bookkeeper.client.api.LastConfirmedAndEntry;

/**
 * 
 *
 */
public class App 
{
    private static final String METADATA_SERVICE_URI = "zk+hierarchical://10.0.2.15:2181/ledgers";

    private static void printCurrentThreadInfo() {
        Thread t = Thread.currentThread();
        System.out.println("Fun: " + Thread.currentThread().getStackTrace()[2].getMethodName() +
                " | 当前线程名字：" + t.getName() + 
                " | 当前线程的优先级别为：" + t.getPriority() +
                " | ID:" + t.getId());
    }

    private static long asynCreateAndWriteLedgerWithClose(boolean close,
            int writeCount,
            int sleepMillis,
            Consumer<Long> consumer) {
        long ledgerId = -1;

        printCurrentThreadInfo();

        try {
            ClientConfiguration clientConfig = new ClientConfiguration();
            clientConfig.setMetadataServiceUri(METADATA_SERVICE_URI);
            BookKeeper bkClient = BookKeeper.forConfig(clientConfig).build();
            CompletableFuture<Long> leadgerIdFuture = bkClient.newCreateLedgerOp()
                .withEnsembleSize(3)
                .withWriteQuorumSize(2)
                .withAckQuorumSize(2)
                .withDigestType(DigestType.MAC)
                .withPassword("some-password".getBytes())
                .execute()
                .thenApplyAsync(w -> {
                    printCurrentThreadInfo();

                    if (null != consumer) {
                        consumer.accept(w.getId());
                    }

                    long lId = -1;
                    try {
                        for (int i = 0; i < writeCount; ++i) {
                            w.append(String.valueOf(i).getBytes());
                            Thread.sleep(sleepMillis);
                        }

                        lId = w.getId();
                        w.close();
                    } catch (InterruptedException | BKException ee) {
                        ee.printStackTrace();
                    }

                    return lId;
                }
                );
                
            ledgerId = leadgerIdFuture.get();

            bkClient.close();
        } catch (InterruptedException | IOException | BKException | ExecutionException e) {
            e.printStackTrace();
        }

        return ledgerId;
    }

    private static long createAndWriteLedgerWithClose(boolean close) {
        long ledgerId = -1;
        
        try {
            ClientConfiguration clientConfig = new ClientConfiguration();
            clientConfig.setMetadataServiceUri(METADATA_SERVICE_URI);
            BookKeeper bkClient = BookKeeper.forConfig(clientConfig).build();
            WriteHandle writer = bkClient.newCreateLedgerOp()
                .withEnsembleSize(3)
                .withWriteQuorumSize(2)
                .withAckQuorumSize(2)
                .withDigestType(DigestType.MAC)
                .withPassword("some-password".getBytes())
                .execute()
                .get();
            writer.append("1".getBytes());
            writer.append("2".getBytes());

            ledgerId = writer.getId();

            if (close) {
                writer.close();
            }

            bkClient.close();
        } catch (InterruptedException | IOException | BKException | ExecutionException e) {
            e.printStackTrace();
        }

        return ledgerId;
    }

    private static void readLedger(long ledgerId) {
        try {
            ClientConfiguration clientConfig = new ClientConfiguration();
            clientConfig.setMetadataServiceUri(METADATA_SERVICE_URI);
            BookKeeper bkClient = BookKeeper.forConfig(clientConfig).build();
            ReadHandle reader = bkClient.newOpenLedgerOp()
                .withLedgerId(ledgerId)
                .withRecovery(false)
                .withDigestType(DigestType.MAC)
                .withPassword("some-password".getBytes())
                .execute()
                .get();

            long lastAdd = reader.readLastAddConfirmed();
            Iterator<LedgerEntry> entries = reader.read(0, lastAdd).iterator();
            while (entries.hasNext()) {
                LedgerEntry entry = entries.next();
                System.out.printf("Successfully read entry | ledger id:%d | entry id:%d | data:%s\n",
                        entry.getLedgerId(),
                        entry.getEntryId(),
                        new String(entry.getEntryBytes()));
            }

            reader.close();

            bkClient.close();
        } catch (InterruptedException | IOException | BKException | ExecutionException e) {
            e.printStackTrace();
        }
    }

    private static void asyncReadLedger(long ledgerId) {
        try {
            ClientConfiguration clientConfig = new ClientConfiguration();
            clientConfig.setMetadataServiceUri(METADATA_SERVICE_URI);
            BookKeeper bkClient = BookKeeper.forConfig(clientConfig).build();
            CompletableFuture<ReadHandle> bkReader = bkClient.newOpenLedgerOp()
                .withLedgerId(ledgerId)
                .withRecovery(false)
                .withDigestType(DigestType.MAC)
                .withPassword("some-password".getBytes())
                .execute()
                .thenApplyAsync(reader -> {
                    System.out.printf("GetLastAddConfirmed: %d\n", reader.getLastAddConfirmed());

                    CompletableFuture<Long> lastAddConfirmedFuture = reader.readLastAddConfirmedAsync();
                    CompletableFuture<Long> thenLastAddConfirmedFuture = lastAddConfirmedFuture.thenApplyAsync(lastAdd -> {
                        printCurrentThreadInfo();
                        try {
                            System.out.printf("readLastAddConfirmed: %d\n", reader.getLastAddConfirmed());

                            Iterator<LedgerEntry> entries = reader.read(0, lastAdd).iterator();
                            while (entries.hasNext()) {
                                LedgerEntry entry = entries.next();
                                System.out.printf("Successfully async read entry | ledger id:%d | entry id:%d | data:%s\n",
                                        entry.getLedgerId(),
                                        entry.getEntryId(),
                                        new String(entry.getEntryBytes()));
                            }
                        } catch (InterruptedException | BKException e) {
                            e.printStackTrace();
                        }

                        return lastAdd; 
                    });

                    thenLastAddConfirmedFuture.join();

                    return reader;
                }
            );


            bkReader.get().close();

            bkClient.close();
        } catch (InterruptedException | IOException | BKException | ExecutionException e) {
            e.printStackTrace();
        }
    }

    private static class WriteThread extends Thread {
        private ReadThread reader;

        public WriteThread() {
            super();
            reader = new ReadThread();
        }

        public void run() {
             asynCreateAndWriteLedgerWithClose(true,
                    100,
                    100,
                    (ledgerId) -> {
                        reader.setLedgerId(ledgerId);
                        reader.start();
                    });
            System.out.printf("Async write ledger is done.");
        }

        public void joinAll() {
            try {
                reader.join();
                this.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    private static class ReadThread extends Thread {
        long currentLedgerId;

        public void setLedgerId(long ledgerId) {
            currentLedgerId = ledgerId;
        }

        public void run() {
            testPollingTailRead();
        }

        private void testReadLastAddConfirmed() {
            try {
                ClientConfiguration clientConfig = new ClientConfiguration();
                clientConfig.setMetadataServiceUri(METADATA_SERVICE_URI);
                BookKeeper bkClient = BookKeeper.forConfig(clientConfig).build();
                ReadHandle reader = bkClient.newOpenLedgerOp()
                    .withLedgerId(currentLedgerId)
                    .withRecovery(false)
                    .withDigestType(DigestType.MAC)
                    .withPassword("some-password".getBytes())
                    .execute()
                    .get();

                long lastAdd = -1;

                for (int i = 0; i < 15; ++i) {
                    // getLastAddConfirmed 是只读取本地存的LastAddConfirmed
                    // readLastAddConfirmed 是从服务端拉取，然后更新本地的LastAddConfirmed
                    long oldLastAdd = reader.getLastAddConfirmed();
                    lastAdd = reader.readLastAddConfirmed();
                    long newLastAdd = reader.getLastAddConfirmed();
                    System.out.printf("Successfully read last add confirmed: %d | old:%d | new:%d\n", lastAdd, oldLastAdd, newLastAdd);
                    Thread.sleep(1000);
                }

                reader.close();

                bkClient.close();
            } catch (InterruptedException | IOException | BKException | ExecutionException e) {
                e.printStackTrace();
            }
        }

        private void testPollingTailRead() {
            try {
                Thread.sleep(2000);

                ClientConfiguration clientConfig = new ClientConfiguration();
                clientConfig.setMetadataServiceUri(METADATA_SERVICE_URI);
                BookKeeper bkClient = BookKeeper.forConfig(clientConfig).build();
                ReadHandle reader = bkClient.newOpenLedgerOp()
                    .withLedgerId(currentLedgerId)
                    .withRecovery(false)
                    .withDigestType(DigestType.MAC)
                    .withPassword("some-password".getBytes())
                    .execute()
                    .get();

                long lastAdd = -1;
                long batchReadSize = 10;
                long nextEntryId = 0;
                long currentLastAdd = reader.getLastAddConfirmed();
                System.out.printf("Successfully xxx | last add:%d \n", currentLastAdd);

                while (!reader.isClosed()) {
                    if (nextEntryId > currentLastAdd) {
                        LastConfirmedAndEntry lastConfAndEntry = reader.readLastAddConfirmedAndEntryAsync(nextEntryId,
                                1000,
                                false)
                            .get();
                        if (null != lastConfAndEntry &&
                                lastConfAndEntry.hasEntry()) {
                            currentLastAdd = lastConfAndEntry.getLastAddConfirmed();
                            LedgerEntry entry = lastConfAndEntry.getEntry();
                            ++nextEntryId;

                            System.out.printf("Successfully async read entry | ledger id:%d | entry id:%d | last add:%d |  data:%s\n",
                                    entry.getLedgerId(),
                                    entry.getEntryId(),
                                    lastConfAndEntry.getLastAddConfirmed(),
                                    new String(entry.getEntryBytes()));
                                } else {
                                    continue;
                                }
                    } else {
                        long currentReadEnd = Math.min(currentLastAdd, nextEntryId + batchReadSize);
                        Iterator<LedgerEntry> entries = reader.read(nextEntryId, currentReadEnd).iterator();
                        while (entries.hasNext()) {
                            LedgerEntry entry = entries.next();
                            System.out.printf("Successfully batch async read entry | ledger id:%d | entry id:%d | data:%s\n",
                                    entry.getLedgerId(),
                                    entry.getEntryId(),
                                    new String(entry.getEntryBytes()));
                        }

                        nextEntryId = currentReadEnd + 1;
                    }
                }

                reader.close();

                bkClient.close();
            } catch (InterruptedException | IOException | BKException | ExecutionException e) {
                e.printStackTrace();
            }
        }
    }

    private static void pollingTailRead() {
        WriteThread writer = new WriteThread();
        writer.start();
        writer.joinAll();
    }

    public static void main(String[] args) {
        /*
        long ledgerId = createAndWriteLedgerWithClose(true);
        readLedger(ledgerId);

        long ledgerId = asynCreateAndWriteLedgerWithClose(true, 2, 100, null);
        asyncReadLedger(ledgerId);
        */

        pollingTailRead();
    }
}
