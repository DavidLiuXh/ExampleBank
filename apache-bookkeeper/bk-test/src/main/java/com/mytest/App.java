package com.mytest;

import java.io.IOException;
import java.util.Iterator;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.CompletableFuture;

import org.apache.bookkeeper.conf.ClientConfiguration;
import org.apache.bookkeeper.client.BookKeeper;
import org.apache.bookkeeper.client.api.LedgerEntry;
import org.apache.bookkeeper.client.api.WriteHandle;
import org.apache.bookkeeper.client.api.ReadHandle;
import org.apache.bookkeeper.client.api.LedgerEntries;
import org.apache.bookkeeper.client.api.DigestType;
import org.apache.bookkeeper.client.api.BKException;

/**
 * 
 *
 */
public class App 
{
    private static final String METADATA_SERVICE_URI = "zk+hierarchical://10.0.2.15:2181/ledgers";

    private static void printCurrentThreadInfo(String msg) {
        Thread t = Thread.currentThread();
        System.out.println("Flag: " + msg + " | 当前线程名字：" + t.getName() + " | 当前线程的优先级别为：" + t.getPriority() + " | ID:" + t.getId());
    }

    private static long asynCreateAndWriteLedgerWithClose(boolean close) {
        long ledgerId = -1;

        try {
            ClientConfiguration clientConfig = new ClientConfiguration();
            clientConfig.setMetadataServiceUri(METADATA_SERVICE_URI);
            BookKeeper bkClient = BookKeeper.forConfig(clientConfig).build();
            CompletableFuture<WriteHandle> writer = bkClient.newCreateLedgerOp()
                .withEnsembleSize(3)
                .withWriteQuorumSize(2)
                .withAckQuorumSize(2)
                .withDigestType(DigestType.MAC)
                .withPassword("some-password".getBytes())
                .execute()
                .whenComplete((w, e) -> {
                System.out.printf("ssssss");
                    if (e != null) {
                        try {
                            w.append("1".getBytes());
                            w.append("2".getBytes());
                        } catch (InterruptedException | org.apache.bookkeeper.client.api.BKException ee) {
                            ee.printStackTrace();
                        }
                    } else {
                        e.printStackTrace();
                    }
                }
                );

            ledgerId = writer.get().getId();
            if (close) {
                writer.get().close();
            }
        } catch (InterruptedException | IOException | org.apache.bookkeeper.client.api.BKException | ExecutionException e) {
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
        } catch (InterruptedException | IOException | org.apache.bookkeeper.client.api.BKException | ExecutionException e) {
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
        } catch (InterruptedException | IOException | org.apache.bookkeeper.client.api.BKException | ExecutionException e) {
            e.printStackTrace();
        }
    }

    private static void asyncReadLedger(long ledgerId) {
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

            printCurrentThreadInfo("Main");

            CompletableFuture<Long> lastAddConfirmedFuture = reader.readLastAddConfirmedAsync();
            lastAddConfirmedFuture.thenApplyAsync(lastAdd -> {
                printCurrentThreadInfo("Future Read Entries");
                try {
                    Iterator<LedgerEntry> entries = reader.read(0, lastAdd).iterator();
                    while (entries.hasNext()) {
                        LedgerEntry entry = entries.next();
                        System.out.printf("Successfully async read entry | ledger id:%d | entry id:%d | data:%s\n",
                                entry.getLedgerId(),
                                entry.getEntryId(),
                                new String(entry.getEntryBytes()));
                    }
                } catch (InterruptedException | org.apache.bookkeeper.client.api.BKException e) {
                    e.printStackTrace();
                }

                return lastAdd; 
            })
            .exceptionally(e -> {
                e.printStackTrace();
                return Long.valueOf(-1);
            });

            reader.close();
        } catch (InterruptedException | IOException | org.apache.bookkeeper.client.api.BKException | ExecutionException e) {
            e.printStackTrace();
        }
    }
    public static void main(String[] args) {
        long ledgerId = createAndWriteLedgerWithClose(true);
        asyncReadLedger(ledgerId);
        //long ledgerId = asynCreateAndWriteLedgerWithClose(true);
        //readLedger(ledgerId);
    }
}