* bk_server.conf 修改
 1. bookiePort
 2. advertisedAddress
 3. httpServerPort
 4. journalDirectories
 5. ledgerDirectories
 6. indexDirectories
 7. metadataServiceUri
* log4j.properties 修改
 1. bookkeeper.root.logger
 2. bookkeeper.log.dir
* bkenv.sh 修改
 1. BOOKE_LOG_DIR
 2. BOOKIE_ROOT_LOGGER
 均与log4j.properties中一致即可, 不然log配置不生效
